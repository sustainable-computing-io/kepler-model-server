FROM --platform=linux/amd64 quay.io/sustainable_computing_io/kepler_model_server_base:latest

WORKDIR /usr/local

RUN mkdir -p /usr/local/src
RUN mkdir -p /usr/local/resource

COPY src/estimate src/estimate
COPY src/server src/server
COPY src/train src/train
COPY src/util src/util
COPY cmd cmd

# port for Model Server
EXPOSE 8100
# port for Online Trainer (TODO: reserved for event-based online training)
EXPOSE 8101
# port for Offline Trainer
EXPOSE 8102

ENTRYPOINT ["python3.8", "cmd/main.py"]