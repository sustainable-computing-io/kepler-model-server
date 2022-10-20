FROM quay.io/sustainable_computing_io/kepler_model_server_base:latest
COPY .. .
ENV MODEL_PATH "../demo/models"
# port for exporter
EXPOSE 9000
CMD [ "python3.8", "demo/demo.py" ]