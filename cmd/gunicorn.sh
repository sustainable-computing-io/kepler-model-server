#!/bin/sh

# Instantiate Gunicorn Instance
launch_gunicorn() {
    FLASK_APP_MODULE=$1
    PORT=$2

    echo "Starting Gunicorn for $FLASK_APP_MODULE on port $PORT"
    gunicorn --workers ${GUNICORN_PROCESSES:-2} \
             --threads ${GUNICORN_THREADS:-4} \
             --bind 0.0.0.0:$PORT \
             --forwarded-allow-ips '*' \
             $APP_MODULE &
}

# Wait for Available Port
# wait_for_open_port() {
#     PORT=$1
#     while ! nc -z localhost $PORT; do
#       echo "Waiting for port $PORT..."
#       sleep 1
#     done
# }

# Model Server
launch_gunicorn "src.server.model_server:app" ${PORT_MODEL_SERVER}

# TODO: Online Trainer
#launch_gunicorn "src.train.online_trainer:app" ${PORT_ONLINE_TRAINER}

# Offline Trainer
launch_gunicorn "src.train.offline_trainer:app" ${PORT_OFFLINE_TRAINER}


# Wait for all ports to be ready
# wait_for_open_port ${PORT_MODEL_SERVER}
# #wait_for_port ${PORT_ONLINE_TRAINER}
# wait_for_open_port ${PORT_OFFLINE_TRAINER}

# Run the main Python script
# echo "Run cmd/main.py script"
# python3.8 cmd/main.py

wait
