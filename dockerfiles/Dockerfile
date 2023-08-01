FROM --platform=linux/amd64 quay.io/sustainable_computing_io/kepler_model_server_base:latest

WORKDIR /usr/local
RUN python3.8 -m pip install py-cpuinfo==9.0.0

RUN mkdir -p /usr/local/src
RUN mkdir -p /usr/local/resource

COPY resource/profiles resource/profiles
COPY src/profile/tool src/profile/tool
COPY src/estimate src/estimate
COPY src/server src/server
COPY src/train src/train
COPY src/util src/util
COPY src/models/default src/models/default
COPY cmd cmd

# port for Model Server
EXPOSE 8100
# port for Online Trainer (TODO: reserved for event-based online training)
EXPOSE 8101
# port for Offline Trainer
EXPOSE 8102

ENTRYPOINT ["python3.8", "cmd/main.py"]