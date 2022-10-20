IMAGE ?= quay.io/sustainable_computing_io/kepler_model_server_demo:latest

build-demo:     ## Build Image for DEMO
	docker build -t $(IMAGE) -f Dockerfile ..
test:
	docker run -p 127.0.0.1:9091:9000/tcp $(IMAGE)