FROM --platform=linux/amd64 quay.io/sustainable_computing_io/kepler_model_server_base:latest

WORKDIR /kepler_model

COPY cmd/ .
COPY src/ .
COPY pyproject.toml .
COPY README.md .
RUN pip install --no-cache-dir .

# port for Model Server
EXPOSE 8100
# port for Online Trainer (TODO: reserved for event-based online training)
EXPOSE 8101
# port for Offline Trainer
EXPOSE 8102

ENTRYPOINT ["kepler-model"]
