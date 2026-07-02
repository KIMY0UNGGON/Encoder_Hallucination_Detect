
import json
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from safetensors.torch import load_file

CKPT_PATH = "model_data/model.safetensors"
TEST_JSON = "상세검색_국회회의록_발언목록_2026-07-01_데이터.json"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LABELS = ["intrinsic", "logical", "extrinsic"]


class GEGLU(nn.Module):
    def __init__(self, in_f,out_f = None, mult=2, dropout= None):
        super().__init__()
        self.up = nn.Linear(in_f, in_f*mult)
        self.down = nn.Linear(in_f*mult//2, out_f) if out_f is not None else None
        self.gelu = nn.GELU()
        self.size = in_f*mult
        self.dropout = nn.Dropout(dropout) if dropout is not None else None
    def forward(self,x):
        x = self.up(x)
        x1 = x[...,:self.size//2]
        x2= x[...,self.size//2:]
        x2 = self.gelu(x2)
        x = x1 * x2
        x = self.down(x) if self.down is not None else x
        x = self.dropout(x) if self.dropout is not None else x
        return x






class CrossAttnLayer(nn.Module):
    """Pre-norm cross-attn:  q = q + CrossAttn(LN(q), LN(kv))."""
    def __init__(self, d, n_heads, dropout=0.1):
        super().__init__()
        self.qn = nn.LayerNorm(d)
        self.kvn = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)

    def forward(self, q, kv, key_padding_mask):
        q_ln = self.qn(q)
        kv_ln = self.kvn(kv)
        a, _ = self.attn(q_ln, kv_ln, kv_ln,
                         key_padding_mask=key_padding_mask, need_weights=False)
        return q + a


class CLSClassifier(nn.Module):
    """순차 co-attention 블록 × n_blocks → [CLS_f, CLS_r] → head → 3 logit.
    각 블록:  역방향 Q=[CLS_r, src], K/V=hyp  →  정방향 Q=[CLS_f, hyp], K/V=[CLS_r, src]."""
    def __init__(self, backbone, hidden, n_labels, dropout, n_heads, sep_id, n_blocks):
        super().__init__()
        self.backbone = backbone
        self.sep_id = sep_id
        self.fwd_layers = nn.ModuleList([CrossAttnLayer(hidden, n_heads, dropout) for _ in range(n_blocks)])
        self.rev_layers = nn.ModuleList([CrossAttnLayer(hidden, n_heads, dropout) for _ in range(n_blocks)])
        self.ffn_norm = nn.LayerNorm(hidden)
        self.ffn = GEGLU(in_f= hidden,out_f = hidden,dropout=dropout)
        self.final_norm = nn.LayerNorm(hidden * 2)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, n_labels)
        )

    def _split(self, h, input_ids, attention_mask):
        """h[B,T,d] → SEP 기준으로 src·hyp 토큰 분리 (padded, valid 마스크 포함)."""
        B = h.shape[0]
        sep = (input_ids == self.sep_id).clone(); sep[:, 0] = False
        srcs, hyps = [], []
        for b in range(B):
            pos = sep[b].nonzero(as_tuple=True)[0]
            n = int(attention_mask[b].sum().item())
            if pos.numel() >= 2:
                s1, s2 = int(pos[0]), int(pos[1])
            elif pos.numel() == 1:
                s1, s2 = int(pos[0]), n
            else:
                s1, s2 = 1, n
            src = h[b, 1:s1]; hyp = h[b, s1 + 1:s2]
            if src.shape[0] == 0: src = h[b, 0:1]
            if hyp.shape[0] == 0: hyp = h[b, 0:1]
            srcs.append(src); hyps.append(hyp)
        src = pad_sequence(srcs, batch_first=True)
        hyp = pad_sequence(hyps, batch_first=True)
        src_valid = torch.zeros(B, src.shape[1], dtype=torch.bool, device=h.device)
        hyp_valid = torch.zeros(B, hyp.shape[1], dtype=torch.bool, device=h.device)
        for b in range(B):
            src_valid[b, :srcs[b].shape[0]] = True
            hyp_valid[b, :hyps[b].shape[0]] = True
        return src, src_valid, hyp, hyp_valid

    def forward(self, input_ids, attention_mask):
        h = self.backbone(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        B = h.shape[0]
        cls_f = cls_r = h[:, :1]
        src, src_v, hyp, hyp_v = self._split(h, input_ids, attention_mask)
        src_pad, hyp_pad = ~src_v, ~hyp_v
        false1 = torch.zeros(B, 1, dtype=torch.bool, device=h.device)
        for i in range(len(self.fwd_layers)):
            qr = torch.cat([cls_r, src], dim=1)                 # 역방향
            qr = self.rev_layers[i](qr, hyp, hyp_pad)
            qr = qr + self.ffn(self.ffn_norm(qr))
            cls_r, src = qr[:, :1], qr[:, 1:]
            qf = torch.cat([cls_f, hyp], dim=1)                 # 정방향
            kv = torch.cat([cls_r, src], dim=1)
            kv_pad = torch.cat([false1, src_pad], dim=1)
            qf = self.fwd_layers[i](qf, kv, kv_pad)
            qf = qf + self.ffn(self.ffn_norm(qf))
            cls_f, hyp = qf[:, :1], qf[:, 1:]
        pooled = torch.cat([cls_f.squeeze(1), cls_r.squeeze(1)], dim=1)   # [B, 2d]
        return self.head(self.final_norm(pooled))                        # [B, 3]


def load_model(ckpt_path=CKPT_PATH, device=DEVICE):
    """model.safetensors → (model, tokenizer, thresholds, cfg).
    텐서 = state_dict, metadata(JSON 문자열) = cfg/thresholds."""
    state = load_file(ckpt_path)
    with safe_open(ckpt_path, framework="pt") as f:
        meta = f.metadata()
    cfg = json.loads(meta["cfg"])
    thresholds = json.loads(meta["thresholds"])

    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    sep_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    backbone = AutoModel.from_pretrained(cfg["backbone"])
    lc = LoraConfig(r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                    lora_dropout=cfg["lora_dropout"],
                    target_modules=cfg["lora_targets"], bias="none")
    backbone = get_peft_model(backbone, lc)

    model = CLSClassifier(backbone, backbone.config.hidden_size, 3, cfg["head_dropout"],
                          n_heads=cfg["n_heads"], sep_id=sep_id, n_blocks=cfg["n_blocks"])
    model.load_state_dict(state)
    model.to(device).eval()
    return model, tok, thresholds, cfg



class return_model:
    def __init__(self):
        self.model, self.tok, self.thresholds, self.cfg = load_model()
        self.device = DEVICE
        self.max_len = self.cfg["max_len"]

    @torch.no_grad()
    def predict(self,source, hypothesis, device=DEVICE, max_len=8192):
        """단일 (source, hypothesis) → {label: (확률, 판정 0/1)}."""
        try:    # 소스만 truncate. 가설이 max_len 보다 길면 실패 → 둘 다 truncate 폴백
            enc = self.tok(source, hypothesis, truncation="only_first",
                    max_length=max_len, return_tensors="pt")
        except Exception:
            enc = self.tok(source, hypothesis, truncation=True,
                    max_length=max_len, return_tensors="pt")
        enc = enc.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=str(device).startswith("cuda")):
            logits = self.model(enc["input_ids"], enc["attention_mask"])
        probs = torch.sigmoid(logits.float()).squeeze(0).cpu()
        return {LABELS[c]: (round(float(probs[c]), 4), int(probs[c] >= self.thresholds[c]))
                for c in range(3)}


if __name__ == "__main__":
    model = return_model()

    with open(TEST_JSON, encoding="utf-8") as f:
        data = json.load(f)    # [{id, passage, Annotation: {SummaryN: 요약문}, Label: {SummaryN: [3]}}]

    print("유형: intrinsic=원문 내용 왜곡 | logical=논리·맥락 오류 | extrinsic=원문에 없는 내용 추가")
    print("출력: 유형별 (확률, 예측판정) | 라벨=정답 [intrinsic, logical, extrinsic]  (1=할루시네이션, 0=정상)")
    total = exact = 0
    for item in data:
        for name, summary in item["Annotation"].items():
            label = item["Label"][name]
            pred = model.predict(item["passage"], summary)
            ok = [pred[k][1] for k in LABELS] == label
            total += 1; exact += ok
            print(f"[id={item['id']}|{name}] {'O' if ok else 'X'} 예측 {pred}  라벨 {label}")
    print(f"exact match: {exact}/{total} ({exact / total * 100:.1f}%)")

