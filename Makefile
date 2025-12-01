DOCKER_IMAGE ?= batch-infer
DOCKER_TAG ?= latest
IMAGE := $(DOCKER_IMAGE):$(DOCKER_TAG)

.PHONY: build run test test-coverage test-image shell

build:
	docker build -t $(IMAGE) .

run: build
	docker run --rm \
		-v "$(PWD):/app" \
		-w /app \
		$(IMAGE) \
		poetry run python -m examples.run_classification

test: build
	docker run --rm \
		-v "$(PWD):/app" \
		-w /app \
		$(IMAGE) \
		poetry run pytest

test-coverage: build
	docker run --rm \
		-v "$(PWD):/app" \
		-w /app \
		$(IMAGE) \
		poetry run pytest --cov=batch_infer --cov-report=term-missing

test-image: build
	docker run --rm \
		$(IMAGE) \
		poetry run pytest

shell: build
	docker run --rm -it \
		-v "$(PWD):/app" \
		-w /app \
		$(IMAGE) \
		/bin/bash
