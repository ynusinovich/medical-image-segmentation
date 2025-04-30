# Makefile for Tumor Segmentation Project

# Build Docker images
build-dev:
	docker build -t medimgseg-dev -f Dockerfile_dev .

build-inference:
	docker build -t medimgseg-inference -f Dockerfile_inference .

# Run Docker containers
run-dev:
	docker run --gpus all -it --rm --shm-size=32g -p 8888:8888 -v $(shell pwd):/app medimgseg-dev

run-inference:
	docker run --gpus all -it --rm --shm-size=32g -p 8000:8000 -v $(shell pwd):/app medimgseg-inference

# Train and Monitor
train:
	pipenv run python scripts/tune_model.py

mlflow-ui:
	pipenv run mlflow ui

# Format code
format:
	pipenv run black .
	pipenv run isort .

# Test Inference API
test-api:
	pipenv run python scripts/test_api_client.py
