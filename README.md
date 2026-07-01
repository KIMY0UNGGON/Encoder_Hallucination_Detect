# Encoder_Hallucination_Detect
인코더 모델을 이용해 본문과 요약문을 확인해 어떤 할루시네이션인지 분류하는 모델 연구입니다
본 모델은 mmbert모델에 AIHUB의 	문서요약 텍스트 데이터를 사용하여 transfer learning을 해 한국어 도메인을 강화한 백본을 사용하고 있습니다.
본 모델에서의 fine-tuning은 방송 콘텐츠 대본 요약 데이터를 수정 하여서 학습하였습니다.
