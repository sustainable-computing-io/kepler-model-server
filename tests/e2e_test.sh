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
	kubectl get pods -n kepler -o yaml
    kubectl logs -n kepler $(get_component exporter) -c estimator
}

get_server_log() {
    kubectl logs -n kepler $(get_component model-server) -c server-api
}

get_db_log(){
    kubectl logs -n kepler model-db -c trainer
    kubectl logs -n kepler model-db
}

wait_for_kepler() {
	local ret=0
	kubectl rollout status ds kepler-exporter -n kepler --timeout 5m || ret=1
	kubectl describe ds -n kepler kepler-exporter || ret=1
	kubectl get pods -n kepler || ret=1

	kubectl logs -n kepler ds/kepler-exporter --all-containers || ret=1
	return $ret
}

wait_for_server() {
    kubectl rollout status deploy kepler-model-server -n kepler --timeout 5m
    wait_for_keyword server "initial pipeline is loaded" "server cannot load initial pipeline"
    wait_for_keyword server "Press CTRL+C to quit" "server has not started yet"
    get_server_log
    kubectl get svc kepler-model-server -n kepler
    kubectl get endpoints kepler-model-server -n kepler
}

wait_for_db() {
    kubectl get po model-db -n kepler
    kubectl wait -n kepler --for=jsonpath='{.status.phase}'=Running pod/model-db --timeout 5m || true
    kubectl describe po model-db -n kepler
    kubectl wait -n kepler --for=jsonpath='{.status.phase}'=Running pod/model-db --timeout 1m
    wait_for_keyword db "Http File Serve Serving" "model-db is not serving"
    get_db_log
}
info() {
	echo "INFO: $*"
}

wait_for_keyword() {
    num_iterations=30
    component=$1
    keyword=$2
    message=$3
	info "Waiting for '${keyword}' from ${component} log"

    for ((i = 0; i < num_iterations; i++)); do
        if grep -q "$keyword" <<< $(get_${component}_log); then
            return
        fi
        sleep 5
    done

    echo "timeout ${num_iterations}s waiting for '${keyword}' from ${component} log"
    echo "Error: $message"

    kubectl get po -n kepler -oyaml

    echo "${component} log:"
    get_${component}_log
    # show all status
    kubectl get po -A
    if [ ! -z ${SERVER} ]; then
        get_server_log
    fi
    if [ ! -z ${ESTIMATOR} ]; then
        get_estimator_log
    fi
    exit 1
}

check_estimator_set_and_init() {
    wait_for_keyword kepler "Model Config NODE_COMPONENTS: {ModelType:EstimatorSidecar" "Kepler should set desired config"
}

restart_kepler() {
    kubectl delete po -n kepler -l app.kubernetes.io/component=exporter
    sleep 5
    get_kepler_log
}

restart_model_server() {
    kubectl delete po -l app.kubernetes.io/component=model-server -n kepler
    sleep 10
    wait_for_server
    restart_kepler
}

test() {
    # set options
    DEPLOY_OPTIONS=$1

    echo ${DEPLOY_OPTIONS}

	for opt in ${DEPLOY_OPTIONS}; do export $opt=true; done

    # patch MODEL_TOPURL environment if DB is not available
    if [ -z ${DB} ]; then

        # train and deploy local modelDB
        kubectl apply -f ${top_dir}/manifests/test/file-server.yaml
        sleep 10
        wait_for_db

        if [ ! -z ${ESTIMATOR} ]; then
            kubectl patch configmap -n kepler kepler-cfm --type merge -p "$(cat ${top_dir}/manifests/test/patch-estimator-sidecar.yaml)"
            kubectl patch ds kepler-exporter -n kepler -p '{"spec":{"template":{"spec":{"containers":[{"name":"estimator","env":[{"name":"MODEL_TOPURL","value":"http://model-db.kepler.svc.cluster.local:8110"}]}]}}}}'
			kubectl get -n kepler ds kepler-exporter -o yaml
        fi
        if [ ! -z ${SERVER} ]; then
            kubectl patch deploy kepler-model-server -n kepler -p '{"spec":{"template":{"spec":{"containers":[{"name":"server-api","env":[{"name":"MODEL_TOPURL","value":"http://model-db.kepler.svc.cluster.local:8110"}]}]}}}}'
            kubectl delete po -n kepler -l app.kubernetes.io/component=model-server
        fi
        kubectl delete po -n kepler -l app.kubernetes.io/component=exporter

    fi

    if [ ! -z ${ESTIMATOR} ]; then
        # with estimator
        if [ ! -z ${TEST} ]; then
            # dummy kepler
            kubectl patch ds kepler-exporter -n kepler --patch-file ${top_dir}/manifests/test/power-request-client.yaml
            if [ ! -z ${SERVER} ]; then
                restart_model_server
            fi
            sleep 1
			wait_for_kepler || {
				kubectl get pods -n kepler -oyaml
				exit 1
			}
            wait_for_keyword kepler Done "cannot get power"
        else
            check_estimator_set_and_init
        fi

        if [ ! -z ${SERVER} ]; then
            # with server
            wait_for_server
            restart_kepler
            get_estimator_log
            sleep 5
            wait_for_keyword estimator "load model from model server" "estimator should be able to load model from server"
        else
            # no server
            get_estimator_log
			sleep 10
			kubectl -n kepler logs ds/kepler-exporter --all-containers
			kubectl -n kepler describe ds/kepler-exporter
			kubectl -n kepler get pods -o yaml

            wait_for_keyword estimator "load model from config" "estimator should be able to load model from config"
        fi
    else
        # no estimator
        if [ ! -z ${SERVER} ]; then
            # with server
            if [ ! -z ${TEST} ]; then 
                # dummy kepler
                kubectl patch ds kepler-exporter -n kepler --patch-file ${top_dir}/manifests/test/model-request-client.yaml
                restart_model_server
                wait_for_keyword kepler Done "cannot get model weight"
            else
                wait_for_server
                restart_kepler
                wait_for_keyword kepler "getWeightFromServer.*core" "kepler should get weight from server"
            fi
        fi
    fi

}

echo "e2e: invoked with: $*"
"$@"
