FROM python:3.10-slim

WORKDIR /usr/local

COPY . /usr/local
RUN pip install --no-cache-dir . && \
    pip cache purge
