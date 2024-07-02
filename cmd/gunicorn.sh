#!/bin/sh

echo "Starting Model Server"
export GUNICORN_PORT=$PORT_MODEL_SERVER
gunicorn --config config/gunicorn_config.py src.server.model_server:app &
echo "Starting Offline Trainer"
export GUNICORN_PORT=$PORT_OFFLINE_TRAINER
gunicorn --config config/gunicorn_config.py src.train.offline_trainer:app &
wait
