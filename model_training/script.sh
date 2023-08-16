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
export ENERGY_SOURCE=${ENERGY_SOURCE:-rapl}
export VERSION=${VERSION-0.6}
export PIPELINE_PREFIX=${PIPELINE_PREFIX-"$(uname)-$(uname -r)-$(uname -m)_"}
export CPE_DATAPATH=${CPE_DATAPATH-"$(pwd)/data"}

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
    kubectl get benchmark $BENCHMARK -n ${BENCHMARK_NS} -ojson > data/${BENCHMARK}.json
}

function collect_idle() {
    docker run --rm -v "$(pwd)"/data:/data --network=host quay.io/sustainable_computing_io/kepler_model_server:v0.6 query -o idle --interval 1000
}

function collect_data() {
    BENCHMARK=$1
    BENCHMARK_NS=$2
    SLEEP_TIME=$3
    kubectl apply -f benchmark/${BENCHMARK}.yaml
    wait_for_benchmark ${BENCHMARK} ${BENCHMARK_NS} ${SLEEP_TIME}
    save_benchmark ${BENCHMARK} ${BENCHMARK_NS}
    
    docker run --rm -v "$(pwd)"/data:/data --network=host quay.io/sustainable_computing_io/kepler_model_server:v0.6 query -i ${BENCHMARK} -o ${BENCHMARK}_kepler_query -s ${PROM_SERVER}|| true
    docker run --rm -v "$(pwd)"/data:/data --network=host quay.io/sustainable_computing_io/kepler_model_server:v0.6 query -i ${BENCHMARK} -o ${BENCHMARK}_cpe_query --metric-prefix cpe -s ${PROM_SERVER}|| true
    kubectl delete -f benchmark/${BENCHMARK}.yaml
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
    docker run --rm -v $CPE_DATAPATH:/data --network=host quay.io/sustainable_computing_io/kepler_model_server:v0.6 train -i ${QUERY_RESPONSE} -p ${PIPELINE_NAME} --isolator min --energy-source ${ENERGY_SOURCE}|| true
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
    collect_data coremark cpe-operator-system 60
    collect_data stressng cpe-operator-system 60
    collect_data parsec cpe-operator-system 60
}

function quick_collect() {
    collect_data sample cpe-operator-system 10
}

function train() {
    train_model coremark_kepler_query coremark_train
    train_model stressng_kepler_query stressng_train
    train_model parsec_kepler_query parsec_train
    train_model coremark_kepler_query,stressng_kepler_query,parsec_kepler_query v${VERSION}_train
}

function quick_train() {
    train_model sample_kepler_query v${VERSION}_sample
}   

function validate() {
    BENCHMARK=$1
    docker run --rm -v "$(pwd)"/data:/data quay.io/sustainable_computing_io/kepler_model_server:v0.6 validate -i ${BENCHMARK}_kepler_query --benchmark ${BENCHMARK}
}

function cleanup() {
    clean_cpe_cr
    clean_deployment || true
    cluster_down
}

"$@"
