build:
	docker build -t batch-infer:latest .

run:
	docker run --rm -v "$PWD/batch_infer:/app/batch_infer" batch-infer:latest

test:
	docker run --rm batch-infer poetry run pytest

test-with-coverage:
	docker run --rm batch-infer poetry run pytest --cov=batch_infer --cov-report=term-missing
