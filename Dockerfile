# model.py 실행용 이미지 — torch 2.10.0 + cuda 13.0 (로컬 학습 환경과 동일)
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
# 이미지의 Debian python 이 PEP 668 로 pip 를 막음 — 컨테이너 전용 환경이라 해제해도 안전
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# 토크나이저를 이미지에 미리 캐시 → 컨테이너 실행 시 네트워크 불필요
RUN python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('jhu-clsp/mmBERT-base')"

COPY model.py .

# 모델 데이터는 런타임에 volume 으로 마운트:
#   /app/model_data/model.safetensors
#   /app/modernbert_after_klue_nli/
#   /app/<TEST_JSON>
CMD ["python", "model.py"]
