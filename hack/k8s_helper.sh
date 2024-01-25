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
# Copyright 2023 The Kepler Contributors
#

set -e

rollout_ns_status() {
	local resources
	resources=$(kubectl get deployments,statefulsets,daemonsets -n=$1 -o name)
	for res in $resources; do
		kubectl rollout status $res --namespace $1 --timeout=10m || die "failed to check status of ${res} inside namespace ${1}"
	done
}

_get_value() {
    res=$1
    namespace=$2
    location=$3
    kubectl get $res -n $namespace -ojson|jq -r $location
}

_get_succeed_condition() {
    resource=$1
    name=$2
    namespace=$3
    if [ "$(kubectl get $resource $name -n $namespace -ojson|jq '.status.conditions | length')" == 0 ]; then
        echo Unknown
    else
        location='.status.conditions|map(select(.type="Succeeded"))[0].status'
        _get_value $resource/$name $namespace $location
    fi
}

_log_completed_pod() {
    local resources
    name=$1
    namespace=$2
    location=".status.phase"
	resources=$(kubectl get pods -n=$namespace -o name)
	for res in $resources; do
        if [ "$res" == "pod/${name}-run-stressng-pod" ]; then
            # get parameters and estimation time
            kubectl logs $res -n $namespace|head
        fi
        echo $res
        if [ "$res" == "pod/${name}-presteps-pod" ]; then
            # get parameters and estimation time
            kubectl logs $res -n $namespace -c step-collect-idle|tail
        else
            kubectl logs $res -n $namespace|tail
        fi
	done
}

wait_for_pipelinerun() {
    resource=pipelinerun
    name=$1
    namespace=default
    
    if kubectl get taskruns|grep ${name}-run-stressng; then
        value=$(_get_succeed_condition $resource $name $namespace)
        while [ "$value" == "Unknown" ] ; 
        do
            echo "Wait for pipeline $name to run workload"
            kubectl get pods
            value=$(_get_succeed_condition $resource $name $namespace)
            if kubectl get pod/${name}-run-stressng-pod |grep Running ; then
                estimate_time_line=$(kubectl logs pod/${name}-run-stressng-pod -c step-run-stressng -n $namespace|grep "Estimation Time (s):")
                estimate_time=$(echo ${estimate_time_line}|awk '{print $4}')
                echo "${estimate_time_line}, sleep"
                sleep ${estimate_time}
                break
            fi
            sleep 60
        done
    fi

    value=$(_get_succeed_condition $resource $name $namespace)
    while [ "$value" == "Unknown" ] ; 
    do
        echo "Wait for pipeline $name to be succeeded"
        kubectl get pods
        sleep 60
        value=$(_get_succeed_condition $resource $name $namespace)
    done

    kubectl get taskrun
    _log_completed_pod $name $namespace
    if [ "$value" == "False" ]; then
        exit 1
    fi
}

"$@"
