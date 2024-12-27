# NOTE: Dockerfile for generating quay.io/kepler_model_server/s3 images

FROM python:3.14.0a3-slim

WORKDIR /usr/local

COPY . /usr/local
RUN pip install --no-cache-dir . && \
    pip cache purge
