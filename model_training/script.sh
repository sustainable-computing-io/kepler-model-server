#!/bin/bash

set -e
# Supported CLUSTER_PROVIDER are kind,microshift
export CLUSTER_PROVIDER=${CLUSTER_PROVIDER:-kind}
export IMAGE_TAG=${IMAGE_TAG:-latest-libbpf}
export KIND_CLUSTER_NAME=${KIND_CLUSTER_NAME:-kind-for-training}
export KIND_REGISTRY_NAME=${KIND_REGISTRY_NAME:-kind-registry-for-training}
export REGISTRY_PORT=${REGISTRY_PORT:-5101}
export IMAGE_REPO=${IMAGE_REPO:-localhost:5101}
export PROM_SERVER=${PROM_SERVER:-http://localhost:9090}
export ENERGY_SOURCE=${ENERGY_SOURCE:-rapl,acpi}
export VERSION=${VERSION-v0.7}
export PIPELINE_PREFIX=${PIPELINE_PREFIX-"std_"}
export CPE_DATAPATH=${CPE_DATAPATH-"$(pwd)/data"}
export ENTRYPOINT_IMG=${ENTRYPOINT_IMG-"quay.io/sustainable_computing_io/kepler_model_server:v0.7"}
export MODEL_PATH=$CPE_DATAPATH

mkdir -p $HOME/bin
export PATH=$HOME/bin:$PATH

mkdir -p data
mkdir -p data/cpe-local-log

if ! [ -x $HOME/bin/kubectl ]
then
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" --insecure
    cp kubectl $HOME/bin/kubectl
    chmod +x $HOME/bin/kubectl
fi

rollout_ns_status() {
	local resources
	resources=$(kubectl get deployments,statefulsets,daemonsets -n=$1 -o name)
	for res in $resources; do
		kubectl rollout status $res --namespace $1 --timeout=10m || die "failed to check status of ${res} inside namespace ${1}"
	done
}

function cluster_up() {
	echo "deploying ${CLUSTER_PROVIDER} cluster"
    pushd custom-cluster
    ./main.sh up
    popd
}

function cluster_down() {
    pushd custom-cluster
   	./main.sh down
    popd
}

function deploy_kepler() {
    kubectl apply -f ./deployment/kepler.yaml
    rollout_ns_status kepler
}

function clean_deployment() {
    kubectl delete -f ./deployment
}

function clean_cpe_cr() {
    kubectl delete -f ./benchmark || true 
    kubectl delete -f ../../utils/yaml/cpe-benchmark-operator.yaml || true
}

function deploy_cpe_operator() {
    docker exec --privileged "${KIND_CLUSTER_NAME}"-control-plane mkdir -p /cpe-local-log
    docker exec --privileged "${KIND_CLUSTER_NAME}"-control-plane chmod 777 /cpe-local-log
    kubectl create -f ./deployment/cpe-operator.yaml
    timeout 60s bash -c 'until kubectl get crd benchmarkoperators.cpe.cogadvisor.io 2>/dev/null; do sleep 1; done'
    rollout_ns_status cpe-operator-system
}

function reload_prometheus() {
    sleep 5
    curl -X POST localhost:9090/-/reload
}

function expect_num() {
    BENCHMARK=$1
    BENCHMARK_NS=$2
    kubectl get benchmark ${BENCHMARK} -n ${BENCHMARK_NS} -oyaml > tmp.yaml
    BENCHMARK_FILE=tmp.yaml
    num=$(cat ${BENCHMARK_FILE}|yq ".spec.repetition")
    if [ -z $num ]; then
        num=1
    fi

    for v in $(cat ${BENCHMARK_FILE}|yq eval ".spec.iterationSpec.iterations[].values | length")
    do
        ((num *= v))
    done
    rm tmp.yaml
    echo $num
}

function wait_for_benchmark() {
    BENCHMARK=$1
    BENCHMARK_NS=$2
    SLEEP_TIME=$3
    EXPECT_NUM=$(expect_num ${BENCHMARK} ${BENCHMARK_NS})
    jobCompleted=$(kubectl get benchmark ${BENCHMARK} -n ${BENCHMARK_NS} -ojson|jq -r .status.jobCompleted)
    echo "Wait for ${EXPECT_NUM} ${BENCHMARK} jobs to be completed, sleep ${SLEEP_TIME}s"
    while [ "$jobCompleted" != "${EXPECT_NUM}/${EXPECT_NUM}" ] ; 
    do  
        sleep ${SLEEP_TIME}
        echo "Wait for ${BENCHMARK} to be completed... $jobCompleted, sleep ${SLEEP_TIME}s"
        jobCompleted=$(kubectl get benchmark ${BENCHMARK} -n ${BENCHMARK_NS} -ojson|jq -r .status.jobCompleted)
    done
    echo "Benchmark job completed"
}

function save_benchmark() {
    BENCHMARK=$1
    BENCHMARK_NS=$2
    kubectl get benchmark $BENCHMARK -n ${BENCHMARK_NS} -ojson > $CPE_DATAPATH/${BENCHMARK}.json
}

function collect_idle() {
    ARGS="-o idle --interval 1000"
    if [ -z "$NATIVE" ]; then
        docker run --rm -v $CPE_DATAPATH:/data --network=host ${ENTRYPOINT_IMG} query ${ARGS}
    else
        python ../cmd/main.py query ${ARGS}|| true
    fi
}

function collect_data() {
    BENCHMARK=$1
    BENCHMARK_NS=$2
    SLEEP_TIME=$3
    kubectl apply -f benchmark/${BENCHMARK}.yaml
    wait_for_benchmark ${BENCHMARK} ${BENCHMARK_NS} ${SLEEP_TIME}
    save_benchmark ${BENCHMARK} ${BENCHMARK_NS}
    ARGS="-i ${BENCHMARK} -o ${BENCHMARK}_kepler_query -s ${PROM_SERVER}"
    if [ -z "$NATIVE" ]; then
        docker run --rm -v $CPE_DATAPATH:/data --network=host ${ENTRYPOINT_IMG} query ${ARGS}|| true
    else
        python ../cmd/main.py query ${ARGS}|| true
    fi
    kubectl delete -f benchmark/${BENCHMARK}.yaml
}

function custom_collect_data() {
    BENCHMARK=$1
    ARGS="-i ${BENCHMARK} -o ${BENCHMARK}_kepler_query -s ${PROM_SERVER}"
    python ../cmd/main.py custom_query ${ARGS}|| true
}

function deploy_prom_dependency(){
    kubectl apply -f deployment/prom-kepler-rbac.yaml
    kubectl apply -f deployment/prom-np.yaml
}

function train_model(){
    QUERY_RESPONSE=$1
    PIPELINE_NAME=${PIPELINE_PREFIX}$2
    echo "input=$QUERY_RESPONSE"
    echo "pipeline=$PIPELINE_NAME"
    echo $CPE_DATAPATH
    ARGS="-i ${QUERY_RESPONSE} -p ${PIPELINE_NAME} --energy-source ${ENERGY_SOURCE}"
    if [ -z "$NATIVE" ]; then
        echo "Train with docker"
        docker run --rm -v $CPE_DATAPATH:/data ${ENTRYPOINT_IMG} train ${ARGS}|| true
    else
        echo "Train natively"
        python ../cmd/main.py train ${ARGS}|| true
    fi
}

function prepare_cluster() {
    cluster_up
    deploy_cpe_operator
    deploy_kepler
    deploy_prom_dependency
    reload_prometheus
}

function collect() {
    collect_idle
    collect_data stressng cpe-operator-system 60
}

function custom_collect() {
    custom_collect_data customBenchmark
}

function quick_collect() {
    collect_data sample cpe-operator-system 10
}

function train() {
    train_model stressng_kepler_query ${VERSION}_stressng
}

function quick_train() {
    train_model sample_kepler_query ${VERSION}_sample
}

function custom_train() {
    train_model customBenchmark_kepler_query ${VERSION}_customBenchmark
}

function validate() {
    BENCHMARK=$1
    ARGS="-i ${BENCHMARK}_kepler_query --benchmark ${BENCHMARK}"
    if [ -z "$NATIVE" -a "${BENCHMARK}" == "customBenchmark" ]; then
        docker run --rm -v $CPE_DATAPATH:/data ${ENTRYPOINT_IMG} custom_validate ${ARGS}
    elif [ -z "$NATIVE" ]; then
        docker run --rm -v $CPE_DATAPATH:/data ${ENTRYPOINT_IMG} validate ${ARGS}
    elif [ "${BENCHMARK}" == "customBenchmark" ]; then
        python ../cmd/main.py custom_validate ${ARGS}|| true
    else
        python ../cmd/main.py validate ${ARGS}|| true
    fi
}

function _export() {
    ID=$1
    OUTPUT=$2
    PUBLISHER=$3
    MAIN_COLLECT_INPUT=$4
    INCLUDE_RAW=$5

    if [ $# -lt 4 ]; then
        echo "need arguements: [machine_id] [path_to_models] [publisher] [benchmark_type]"
        exit 2
    fi

    PIPELINE_NAME=${PIPELINE_PREFIX}${VERSION}_${MAIN_COLLECT_INPUT}
    VALIDATE_INPUT="${MAIN_COLLECT_INPUT}_kepler_query"
    ARGS="--id ${ID} -p ${PIPELINE_NAME}  -i ${VALIDATE_INPUT} --benchmark ${MAIN_COLLECT_INPUT} --version ${VERSION} --publisher ${PUBLISHER} ${INCLUDE_RAW}"
    echo "${ARGS}"
    if [ -z "$NATIVE" ]; then
        docker run --rm -v $CPE_DATAPATH:/data -v ${OUTPUT}:/models ${ENTRYPOINT_IMG} export ${ARGS} -o /models 
    else
        python ../cmd/main.py export ${ARGS} -o ${OUTPUT}
    fi
}

function export() {
    _export $1 $2 $3 $4
}

function export_with_raw() {
    _export $1 $2 $3 $4 "--include-raw true"
}

function cleanup() {
    clean_cpe_cr
    clean_deployment || true
    cluster_down
}

"$@"
