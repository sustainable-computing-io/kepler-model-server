export IMAGE_REGISTRY ?= quay.io/sustainable_computing_io
IMAGE_NAME := kepler-model-server
IMAGE_VERSION := 0.6

IMAGE := $(IMAGE_REGISTRY)/$(IMAGE_NAME):v$(IMAGE_VERSION)
TEST_IMAGE := $(IMAGE)-test

CTR_CMD :=$(or $(shell which podman 2>/dev/null), $(shell which docker 2>/dev/null))
CTR_CMD = docker

DOCKERFILES_PATH := "./dockerfiles"

build:
	$(CTR_CMD) build --platform linux/amd64 -t $(IMAGE) -f $(DOCKERFILES_PATH)/Dockerfile .

build-test:
	$(CTR_CMD) build --platform linux/amd64 -t $(TEST_IMAGE) -f $(DOCKERFILES_PATH)/Dockerfile.test .

exec-test:
	$(CTR_CMD) run --platform linux/amd64 -it $(TEST_IMAGE) /bin/bash

test-pipeline:
	$(CTR_CMD) run --platform linux/amd64 -i $(TEST_IMAGE) /bin/bash -c "python3.8 -u ./tests/pipeline_test.py"

# test collector --> estimator
run-estimator:
	$(CTR_CMD) run -d --platform linux/amd64  -p 8100:8100 --name estimator $(TEST_IMAGE) python3.8 src/estimate/estimator.py

run-collector-client:
	$(CTR_CMD) exec estimator /bin/bash -c "while [ ! -S "/tmp/estimator.sock" ]; do sleep 1; done; python3.8 -u ./tests/estimator_power_request_test.py"

clean-estimator:
	$(CTR_CMD) stop estimator
	$(CTR_CMD) rm estimator

test-estimator: run-estimator run-collector-client clean-estimator

# test estimator --> model-server
run-model-server:
	$(CTR_CMD) run -d --platform linux/amd64  -p 8100:8100 --name model-server $(TEST_IMAGE) python3.8 src/server/model_server.py

run-estimator-client:
	$(CTR_CMD) exec model-server /bin/bash -c "python3.8 -u ./tests/estimator_model_request_test.py"

clean-model-server:
	@$(CTR_CMD) stop model-server
	@$(CTR_CMD) rm model-server

test-model-server: run-model-server run-estimator-client clean-model-server

# test offline trainer
run-offline-trainer:
	$(CTR_CMD) run -d --platform linux/amd64  -p 8102:8102 --name offline-trainer $(TEST_IMAGE) python3.8 src/train/offline_trainer.py
	sleep 5

run-offline-trainer-client:
	$(CTR_CMD) exec offline-trainer /bin/bash -c "python3.8 -u ./tests/offline_trainer_test.py"

clean-offline-trainer:
	@$(CTR_CMD) stop offline-trainer
	@$(CTR_CMD) rm offline-trainer

test-offline-trainer: run-offline-trainer run-offline-trainer-client clean-offline-trainer

test: build-test test-pipeline test-estimator test-model-server test-offline-trainer