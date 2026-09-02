.PHONY: data train serve test up

data:
	python scripts/generate_sample_data.py --good 200 --defect 30

train:
	python -m src.train --data data/raw/good --epochs 40

serve:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

test:
	pytest tests/ -q

up:
	docker compose up -d --build
