FROM python:3.8-slim

WORKDIR /usr/local

COPY s3-pusher.py /usr/local

RUN pip install boto3 ibm-cos-sdk

ENTRYPOINT ["python3.8", "s3-pusher.py"]