#!/usr/bin/env bash
#
# This file is part of the Kepler project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Copyright 2022 The Kepler Contributors
#

# Get the directory of the currently executing script_
set -ex

top_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"/..

echo "Top location: $top_dir"

get_component() {
    component=$1
    kubectl get po -n kepler -l app.kubernetes.io/component=${component} -oname
}

get_kepler_log() {
    wait_for_kepler
    kubectl logs -n kepler $(get_component exporter) -c kepler-exporter
}

get_estimator_log() {
    wait_for_kepler
    kubectl logs -n kepler $(get_component exporter) -c estimator
}

get_server_log() {
    kubectl logs -n kepler $(get_component model-server) -c server-api
}

wait_for_kepler() {
    kubectl rollout status ds kepler-exporter -n kepler --timeout 5m
    kubectl describe ds -n kepler kepler-exporter
    kubectl get po -n kepler
}

wait_for_server() {
    kubectl rollout status deploy kepler-model-server -n kepler --timeout 5m
    wait_for_keyword server "initial pipeline is loaded" "server cannot load initial pipeline"
    wait_for_keyword server "Press CTRL+C to quit" "server has not started yet"
}

wait_for_keyword() {
    num_iterations=10
    component=$1
    keyword=$2
    message=$3
    for ((i = 0; i < num_iterations; i++)); do
        if grep -q "$keyword" <<< $(get_${component}_log); then
            return
        fi
        kubectl get po -n kepler -oyaml
        sleep 2
    done
    echo "timeout ${num_iterations}s waiting for '${keyword}' from ${component} log"
    echo "Error: $message"

    echo "${component} log:"
    get_${component}_log
    # show all status
    kubectl get po -A
    exit 1
}

check_estimator_set_and_init() {
    wait_for_keyword kepler "Model Config NODE_COMPONENTS: {ModelType:EstimatorSidecar" "Kepler should set desired config"
}

restart_model_server() {
    kubectl delete po  -l app.kubernetes.io/component=model-server -n kepler
    wait_for_server
}

test() {
    # set options
    DEPLOY_OPTIONS=$1

    echo ${DEPLOY_OPTIONS}

    for opt in ${DEPLOY_OPTIONS}; do export $opt=true; done;

    if [ ! -z ${ESTIMATOR} ]; then
        # with estimator
        if [ ! -z ${TEST} ]; then
            kubectl patch ds kepler-exporter -n kepler --patch-file ${top_dir}/manifests/test/power-request-client.yaml
            if [ ! -z ${SERVER} ]; then
                restart_model_server
            fi
            sleep 1
            wait_for_kepler
            wait_for_keyword kepler Done "cannot get power"
        else
            check_estimator_set_and_init
        fi

        if [ ! -z ${SERVER} ]; then
            wait_for_server
            wait_for_keyword estimator "load model from model server" "estimator should be able to load model from server"
        fi
    else
        # no estimator
        if [ ! -z ${SERVER} ]; then
            if [ ! -z ${TEST} ]; then 
                kubectl patch ds kepler-exporter -n kepler --patch-file ${top_dir}/manifests/test/model-request-client.yaml
                restart_model_server
                sleep 1
                wait_for_kepler
                wait_for_keyword kepler Done "cannot get model weight"
            fi
        fi
    fi

}

"$@"