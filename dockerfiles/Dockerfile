FROM quay.io/sustainable_computing_io/kepler_model_server_base:v0.7.12

WORKDIR /kepler_model
ENV PYTHONPATH=/kepler_model

COPY pyproject.toml .
COPY README.md .
COPY cmd/ cmd/
COPY src/ src/

RUN pip install --no-cache-dir .

# port for Model Server
EXPOSE 8100
# port for Online Trainer (TODO: reserved for event-based online training)
EXPOSE 8101
# port for Offline Trainer
EXPOSE 8102

ENTRYPOINT ["bash", "-c"]
CMD ["kepler-model"]
