FROM python:3.10-slim

WORKDIR /usr/local

COPY . /usr/local

RUN pip install boto3 ibm-cos-sdk

ENTRYPOINT ["python3.10"]
