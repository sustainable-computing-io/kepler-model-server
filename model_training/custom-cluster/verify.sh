#!/bin/bash
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
# Copyright 2023 The Kepler Contributors
#
set -x

NAMESPACE=${NAMESPACE-"monitoring"}

function rollout_status() {
    kubectl rollout status $1 --namespace $2 --timeout=5m || {
        echo "fail to check status of ${1} inside namespace ${2}"
        exit 1
    }
}

function verify_bcc() {
    # basic check for bcc
    if [ $(dpkg -l | grep bcc | wc -l) == 0 ]; then
        echo "no bcc package found"
        exit 1
    fi
}

function verify_cluster() {
    # basic check for k8s cluster info
    if [ $(kubectl cluster-info) !=0 ]; then
        echo "fail to get the cluster-info"
        exit 1
    fi

    # check k8s system pod is there...
    if [ $(kubectl get pods --all-namespaces | wc -l) == 0 ]; then
        echo "it seems k8s cluster is not started"
        exit 1
    fi

    # check rollout status
    resources=$(
        kubectl get deployments --namespace=$NAMESPACE -o name
        kubectl get statefulsets --namespace=$NAMESPACE -o name
    )
    for res in $resources; do
        rollout_status $res $NAMESPACE
    done
}

function main() {
    # verify the deployment of cluster
    case $1 in
    bcc)
        verify_bcc
        ;;
    cluster)
        verify_cluster
        ;;
    *)
        verify_bcc
        verify_cluster
        ;;
    esac
}
main $1
