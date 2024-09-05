IMAGE_REGISTRY ?= quay.io/sustainable_computing_io
IMAGE_NAME := kepler_model_server
IMAGE_VERSION ?= v$(shell cat VERSION)
KEPLER_IMAGE_NAME := kepler
KEPLER_IMAGE_VERSION ?= release-0.7.11

IMAGE ?= $(IMAGE_REGISTRY)/$(IMAGE_NAME):$(IMAGE_VERSION)
KEPLER_IMAGE ?=  $(IMAGE_REGISTRY)/$(KEPLER_IMAGE_NAME):$(KEPLER_IMAGE_VERSION)
BASE_IMAGE ?= $(IMAGE_REGISTRY)/$(IMAGE_NAME)_base:$(IMAGE_VERSION)
TEST_IMAGE := $(IMAGE)-test

CTR_CMD ?= docker
PYTHON  := python3.10

DOCKERFILES_PATH := ./dockerfiles
MODEL_PATH := ${PWD}/tests/models

.PHONY: fmt
fmt:
	@hatch fmt -f

.PHONY: lint
lint:
	@hatch run pymarkdownlnt scan -r .
	@hatch run yamllint -c ./.yamllint.yaml --strict .

.PHONY: build
build:
	$(CTR_CMD) build -t $(IMAGE) -f $(DOCKERFILES_PATH)/Dockerfile .

.PHONY: build-base
build-base:
	$(CTR_CMD) build -t $(BASE_IMAGE) -f $(DOCKERFILES_PATH)/Dockerfile.base .

.PHONY: build-test-nobase
build-test-nobase:
	$(CTR_CMD) build -t $(TEST_IMAGE) -f $(DOCKERFILES_PATH)/Dockerfile.test-nobase .

.PHONY: build-test
build-test:
	$(CTR_CMD) build -t $(TEST_IMAGE) -f $(DOCKERFILES_PATH)/Dockerfile.test .

.PHONY: push
push:
	$(CTR_CMD) push $(IMAGE)

.PHONY: push-base
push-base:
	$(CTR_CMD) push $(BASE_IMAGE)

.PHONY: push-test
push-test:
	$(CTR_CMD) push $(TEST_IMAGE)

exec-test:
	$(CTR_CMD) run --platform linux/amd64 -it $(TEST_IMAGE) /bin/bash

test-pipeline:
	mkdir -p ${MODEL_PATH}
	$(CTR_CMD) run --rm --platform linux/amd64 \
		-v ${MODEL_PATH}:/mnt/models -i \
		$(TEST_IMAGE) \
		hatch run test -vvv -s ./tests/pipeline_test.py

# test collector --> estimator
run-estimator:
	$(CTR_CMD) run --rm -d --platform linux/amd64 \
		-e "MODEL_TOPURL=http://localhost:8110" \
		-v ${MODEL_PATH}:/mnt/models \
		-p 8100:8100 \
		--name estimator \
		$(TEST_IMAGE) \
		/bin/bash -c "$(PYTHON) tests/http_server.py & sleep 5 && estimator --log-level debug"

run-estimator-with-test-spec:
	$(CTR_CMD) run --rm -d --platform linux/amd64 \
		--name estimator \
		$(TEST_IMAGE) \
		/bin/bash -c "estimator --machine-spec tests/data/machine/spec.json --log-level debug"

run-collector-client:
	$(CTR_CMD) exec estimator /bin/bash -c \
		"while [ ! -S "/tmp/estimator.sock" ]; do \
			sleep 1; \
		done; \
		hatch run test -vvv -s ./tests/estimator_power_request_test.py"

clean-estimator:
	@$(CTR_CMD) logs estimator
	@$(CTR_CMD) stop estimator
	@$(CTR_CMD) rm estimator || true

test-estimator: run-estimator run-collector-client clean-estimator

run-model-server:
	$(CTR_CMD) run --rm -d --platform linux/amd64 \
		-e "MODEL_TOPURL=http://localhost:8110" \
		-v ${MODEL_PATH}:/mnt/models \
		-p 8100:8100 \
		--name model-server $(TEST_IMAGE) \
		/bin/bash -c "$(PYTHON) tests/http_server.py & sleep 10 && model-server --log-level debug"; \
	while ! $(CTR_CMD) logs model-server 2>&1 | grep -q 'Running on all'; do \
		echo "... waiting for model-server to serve";  sleep 5; \
	done

run-estimator-client:
	$(CTR_CMD) exec model-server \
		hatch run test -vvv -s ./tests/estimator_model_request_test.py

clean-model-server:
	@$(CTR_CMD) logs model-server
	@$(CTR_CMD) stop model-server
	@$(CTR_CMD) rm model-server || true

test-model-server: \
	run-model-server \
	run-estimator-client \
	clean-model-server

# test offline trainer
run-offline-trainer:
	$(CTR_CMD) run -d --rm --platform linux/amd64  \
		-p 8102:8102 \
		--name offline-trainer \
		$(TEST_IMAGE) \
		offline-trainer
	sleep 5

run-offline-trainer-client:
	$(CTR_CMD) exec offline-trainer \
		hatch run test -vvv -s ./tests/offline_trainer_test.py

clean-offline-trainer:
	@$(CTR_CMD) stop offline-trainer

test-offline-trainer: \
	run-offline-trainer \
	run-offline-trainer-client \
	clean-offline-trainer

# test model server select
create-container-net:
	@$(CTR_CMD) network create kepler-model-server-test

run-model-server-with-db:
	$(CTR_CMD) run -d --platform linux/amd64 \
		--network kepler-model-server-test \
		-p 8100:8100 \
		--name model-server $(TEST_IMAGE) \
		model-server
	while ! $(CTR_CMD) logs model-server 2>&1 | grep -q 'Running on all'; do \
		echo "... waiting for model-server to serve";  sleep 5; \
	done

run-estimator-with-model-server:
	$(CTR_CMD) run -d --platform linux/amd64 \
		--network kepler-model-server-test \
		-e "MODEL_SERVER_ENABLE=true" \
		-e "MODEL_SERVER_URL=http://model-server:8100" \
		--name estimator $(TEST_IMAGE) \
		/bin/bash -c "estimator --machine-spec tests/data/machine/spec.json"

clean-container-net:
	@$(CTR_CMD) network rm kepler-model-server-test

run-select-client:
	$(CTR_CMD) exec model-server \
		hatch run test -vvv -s ./tests/model_select_test.py

test-model-server-select: create-container-net run-model-server-with-db run-select-client clean-model-server clean-container-net

test-model-server-estimator-select: create-container-net run-model-server-with-db run-estimator-with-model-server run-collector-client clean-estimator clean-model-server clean-container-net

test: \
	build-test \
	test-pipeline \
	test-estimator \
	test-model-server \
	test-offline-trainer \
	test-model-server-select \
	test-model-server-estimator-select

# set image
set-image:
	@cd ./manifests/base && kustomize edit set image kepler_model_server=$(IMAGE)
	@cd ./manifests/server && kustomize edit set image kepler_model_server=$(IMAGE)
	@cd ./manifests/kepler && kustomize edit set image kepler=$(KEPLER_IMAGE)

# deploy
_deploy:
	@$(MAKE) set-image
	@kustomize build ./manifests/base|kubectl apply -f -

# print
_print:
	@$(MAKE) set-image
	@kustomize build ./manifests/base|cat

cleanup:
	kustomize build manifests/base|kubectl delete -f -

deploy:
	@chmod +x ./manifests/set.sh
	@./manifests/set.sh "${OPTS}"
	@$(MAKE) _deploy

manifest:
	@chmod +x ./manifests/set.sh
	@./manifests/set.sh "${OPTS}"
	@$(MAKE) _print

e2e-test:
	chmod +x ./tests/e2e_test.sh
	./tests/e2e_test.sh test "${OPTS}"

patch-power-request-client:
	kubectl patch ds kepler-exporter -n kepler --patch-file ./manifests/test/power-request-client.yaml

patch-model-request-client:
	kubectl patch ds kepler-exporter -n kepler --patch-file ./manifests/test/model-request-client.yaml
