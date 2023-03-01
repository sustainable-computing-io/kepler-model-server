#!/bin/bash

if [ -z ${PROMETHEUS_NAMESPACE} ]; then
    PROMETHEUS_NAMESPACE="monitoring"
fi

if [ -z ${PROMEHEUS_POD} ]; then
    PROMEHEUS_POD="prometheus-k8s-0"
fi

if [ -z ${PROMEHEUS_PORT} ]; then
    PROMEHEUS_PORT="9090"
fi

if [ -z ${WORKLOAD_NAMESPACE} ]; then
    WORKLOAD_NAMESPACE="default"
fi

deploy_cpe() {
    kubectl apply -f tool/cpe_operator.yaml
    kubectl rollout status deploy cpe-operator-controller-manager -n cpe-operator-system --timeout 300s
    kubectl rollout status deploy cpe-operator-cpe-parser -n cpe-operator-system --timeout 300s
}

deploy_benchmark_operator() {
    kubectl apply -f tool/benchmark_operator.yaml
}

wait_for_complete() {
    jobCompleted=$(kubectl get benchmark coremark -ojson|jq .status.jobCompleted|tr -d '"')
    echo "Wait for coremark to be completed"
    while [ "$jobCompleted" != "12/12" ] ; 
    do  
        sleep 60
        echo "Wait for coremark to be completed... $jobCompleted, sleep 1m"
        jobCompleted=$(kubectl get benchmark coremark -ojson|jq .status.jobCompleted|tr -d '"')
    done
    echo "Benchmark job completed"
}

deploy_benchmark() {
    kubectl apply -f tool/benchmark.yaml
}

port_forward() {
    kubectl port-forward -n ${PROMETHEUS_NAMESPACE} pods/${PROMEHEUS_POD} 30090:${PROMEHEUS_PORT}&
}

save() {
    port_forward
    kubectl get benchmark coremark -oyaml > tmp.yaml
    resultId=$(md5sum tmp.yaml|awk '{ print $1 }')
    resultPath="../../resource/query_response/${resultId}"
    mkdir -p ${resultPath}
    mv tmp.yaml "${resultPath}/cr.yaml"
    python3 ./tool/query.py "${resultPath}/cr.yaml"
    kill $(pidof kubectl)
    echo "Results are saved in ${resultPath}"
    git checkout -b ${resultId}
    git add ${resultPath}
    git commit -m "contribute profiling response ${resultId}" -s
}

push() {
    git push -u origin $(git branch --show-current)
}

cleanup() {
    kubectl delete -f tool/benchmark.yaml
    kubectl delete -f tool/benchmark_operator.yaml
    kubectl delete -f tool/cpe_operator.yaml
}

check_workload() {
    for namespace in $WORKLOAD_NAMESPACE; do
        count=$(kubectl get po -A|awk '{ print $1 }'|grep ${namespace}|wc -l)
        if [ "$count" != 0 ] ; then
            kubectl get po -n ${namespace}
            echo >&2 "Fatal error: some workload running in ${namespace}"
            exit 2
        fi
    done
    echo "No workload is running in ${WORKLOAD_NAMESPACE}. Ready to run benchmark"
}

run() {
    deploy_cpe
    deploy_benchmark_operator
    deploy_benchmark
    wait_for_complete
    save
    cleanup
}

"$@"