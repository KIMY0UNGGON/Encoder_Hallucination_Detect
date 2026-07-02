# Encoder_Hallucination_Detect
인코더 모델을 이용해 본문과 요약문을 확인해 어떤 할루시네이션인지 분류하는 모델 연구입니다
본 모델은 mmbert모델에 AIHUB의 	문서요약 텍스트 데이터를 사용하여 transfer learning을 해 한국어 도메인을 강화한 백본을 사용하고 있습니다.
본 모델에서의 fine-tuning은 방송 콘텐츠 대본 요약 데이터의 c_event와 culture의 데이터들을 수정 하여서 학습하였습니다.

# 할루시네이션 분류

 현재 본문을 요약할때 AI가 생성하는 할루시네이션을 3가지 종류로 분류

## 1. intrinsic 할루시네이션(내재적 환각)
 본문에 있는 내용이 왜곡되어 요약문에 반영되는 할루시네이션
 ex) 본문에는 10명이 취침을 하고 있다. 라고 적혀있지만 요약문에는 7명이 취침을 하고 있다라고 적혀있는경우

## 2. logical 할루시네이션(논리적 환각)
 요약문에서의 내용이 논리적으로 틀린 내용이 적힌것이다.
 ex) "제219회 국회 본회의에서 회사정리법중개정법률안과 섭외사법개정법률안이 법제사법위원회 수정안대로 가결되었다. 이 의결은 국제사법 분야의 제도 정비가 완전히 마무리되었음을 보여주는 동시에, 그 어떤 제도 정비도 전혀 이루어지지 않았음을 명백히 입증한다." 와 같이 논리적으로 맞지 않는 요약문

## 3. extrinsic 할루시네이션(외재적 환각)
 요약문에 본문에 없는 내용이 들어가는 경우.
 ex) 본문에 없는 출처가 요약문에 들어가는 경우
 
# 모델 구조 
원문(source)·요약문(hypothesis) 쌍을 입력받아 3유형 할루시네이션
(**intrinsic** 원문 내용 왜곡 / **logical** 논리·맥락 오류 / **extrinsic** 원문에 없는 내용 추가)을
multi-label로 판정하는 인코더 기반 분류기.

```mermaid
flowchart TD
    IN["원문 source + 요약문 hypothesis"] --> TOK["mmBERT Tokenizer<br/>CLS + source + SEP + hypothesis + SEP<br/>(max_len 8192, source만 truncate)"]
    TOK --> BB["ModernBERT 백본<br/>KLUE-NLI 파인튜닝 + LoRA (r=16, α=32)<br/>hidden 768"]
    BB --> SPLIT["SEP 위치 기준 토큰 분리"]
    SPLIT --> CLS0["CLS 토큰 → CLS_f, CLS_r 초기화"]
    SPLIT --> SRC["source 토큰"]
    SPLIT --> HYP["hypothesis 토큰"]

    subgraph BLOCK["순차 Co-Attention 블록 × n_blocks (pre-norm)"]
        direction TB
        REV["역방향 Cross-Attn<br/>Q = CLS_r ⊕ src / K·V = hyp"] --> RFFN["잔차 + 공유 GeGLU-FFN<br/>768 → 1536 → GEGLU(split·gate) → 768"]
        RFFN --> FWD["정방향 Cross-Attn<br/>Q = CLS_f ⊕ hyp / K·V = CLS_r ⊕ src"]
        FWD --> FFFN["잔차 + 공유 GeGLU-FFN (동일 가중치)"]
        FFFN -.->|"갱신된 hyp = 다음 블록 역방향의 K/V"| REV
    end

    CLS0 --> BLOCK
    SRC --> BLOCK
    HYP --> BLOCK

    BLOCK --> POOL["readout: CLS_f ⊕ CLS_r concat (1536)"]
    POOL --> HEAD["LayerNorm → Dropout → Linear(1536 → 3)"]
    HEAD --> OUT["sigmoid + 클래스별 threshold<br/>intrinsic / logical / extrinsic (0/1)"]
```

## 구성 요소

| 구성 | 값 |
|---|---|
| 백본 | ModernBERT (mmBERT-base → KLUE-NLI 파인튜닝) |
| 어댑터 | LoRA r=16, α=32, targets: Wqkv·Wo·Wi (백본 동결) |
| Cross-Attention | pre-norm, 12 heads, 역방향 → 정방향 순차 co-attention |
| 공유 FFN | GeGLU (768 → 1536 → split·gate → 768), 블록 내 정·역 공유 |
| Readout | CLS 정방향·역방향 2토큰 concat → Linear → 3 logit |
| 손실 | BCEWithLogits (multi-label) |
| 추론 threshold | validation F1 최적값 (체크포인트 metadata 에 저장) |

## 핵심 설계

- **양방향 정보 교환**: 정방향이 갱신한 hyp 토큰이 다음 블록 역방향의 K/V가 되어,
  블록을 쌓을수록 역방향도 정방향 결과를 본다.
- **CLS 2개 readout**: CLS_f(요약→원문 대조 요약)와 CLS_r(원문→요약 대조 요약)을
  concat 해 방향별 근거 신호를 분리 유지.
- **가중치 로드**: 체크포인트(safetensors)에 state_dict + cfg/threshold(JSON metadata)가
  함께 저장돼 파일 하나로 완전 복원.
