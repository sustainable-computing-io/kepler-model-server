#!/usr/bin/env bash

set -eu -o pipefail

# NOTE: assumes that the project root is one level up
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
declare -r PROJECT_ROOT

declare -r TMP_DIR="$PROJECT_ROOT/tmp"
declare -r LOCAL_DEV_CLUSTER_DIR=${LOCAL_DEV_CLUSTER_DIR:-"$TMP_DIR/local-dev-cluster"}
declare -r DEPOYMENT_DIR=${DEPOYMENT_DIR:-"$PROJECT_ROOT/model_training/deployment"}
declare -r CPE_BENCHMARK_DIR=${CPE_BENCHMARK_DIR:-"$PROJECT_ROOT/model_training/cpe_benchmark"}
declare -r LOCAL_DEV_CLUSTER_VERSION=${LOCAL_DEV_CLUSTER_VERSION:-"main"}

declare KIND_CLUSTER_NAME=${KIND_CLUSTER_NAME:-"kind-for-training"}
declare KIND_REGISTRY_NAME=${KIND_REGISTRY_NAME:-"kind-registry-for-training"}

declare PROM_SERVER=${PROM_SERVER:-"http://localhost:9090"}
declare ENERGY_SOURCE=${ENERGY_SOURCE:-"intel_rapl,acpi"}

declare VERSION=${VERSION:-"latest"}
declare PIPELINE_PREFIX=${PIPELINE_PREFIX:-"std_"}

declare DATAPATH=${DATAPATH:-"$(pwd)/data"}
declare ENTRYPOINT_IMG=${ENTRYPOINT_IMG:-"quay.io/sustainable_computing_io/kepler_model_server:$VERSION"}
declare PROMETHEUS_ENABLE=${PROMETHEUS_ENABLE:-"true"}
declare TEKTON_ENABLE=${TEKTON_ENABLE:-"true"}

declare NATIVE=${NATIVE:-"true"}

clone_local_dev_cluster() {
  [[ -d "$LOCAL_DEV_CLUSTER_DIR" ]] && {
    echo "using local local-dev-cluster"
    return 0
  }

  echo "downloading local-dev-cluster"
  git clone -b "$LOCAL_DEV_CLUSTER_VERSION" \
    https://github.com/sustainable-computing-io/local-dev-cluster.git \
    --depth=1 \
    "$LOCAL_DEV_CLUSTER_DIR"
  return $?
}

rollout_ns_status() {
  local ns="$1"
  shift 1
  local resources=""
  resources=$(kubectl get deployments,statefulsets,daemonsets -n="$ns" -o name)
  for res in $resources; do
    kubectl rollout status "$res" --namespace "$ns" --timeout=10m || {
      echo "failed to check status of $res inside namespace $ns"
      return 1
    }
  done
  return 0
}

cluster_up() {
  clone_local_dev_cluster
  cd "$LOCAL_DEV_CLUSTER_DIR"
  "$LOCAL_DEV_CLUSTER_DIR/main.sh" up
  cd "$PROJECT_ROOT/model_training"
}

cluster_down() {
  cd "$LOCAL_DEV_CLUSTER_DIR"
  "$LOCAL_DEV_CLUSTER_DIR/main.sh" down
}

deploy_kepler() {
  kubectl apply -f "$DEPOYMENT_DIR"/kepler.yaml
  rollout_ns_status kepler
}

clean_deployment() {
  kubectl delete -f "$DEPOYMENT_DIR"
}

clean_cpe_cr() {
  kubectl delete -f "$CPE_BENCHMARK_DIR" || true
  kubectl delete -f "$DEPOYMENT_DIR"/cpe-operator.yaml || true
}

deploy_cpe_operator() {
  docker exec --privileged "$KIND_CLUSTER_NAME"-control-plane mkdir -p /cpe-local-log
  docker exec --privileged "$KIND_CLUSTER_NAME"-control-plane chmod 777 /cpe-local-log
  kubectl create -f "$DEPOYMENT_DIR"/cpe-operator.yaml
  timeout 60s bash -c 'until kubectl get crd benchmarkoperators.cpe.cogadvisor.io 2>/dev/null; do sleep 1; done'
  rollout_ns_status cpe-operator-system
}

reload_prometheus() {
  sleep 5
  curl -X POST localhost:9090/-/reload
}

expect_num() {
  local benchmark="$1"
  local benchmark_ns="$2"
  shift 2
  kubectl get benchmark "$benchmark" -n "$benchmark_ns" -oyaml >tmp.yaml
  local benchmark_file="tmp.yaml"
  num=$(yq ".spec.repetition" <"$benchmark_file")
  [[ -z "$num" ]] && num=1

  for v in $(yq eval ".spec.iterationSpec.iterations[].values | length" <"$benchmark_file"); do
    ((num *= v))
  done
  rm "tmp.yaml"
  echo "$num"
}

wait_for_benchmark() {
  local benchmark="$1"
  local benchmark_ns="$2"
  local sleep_time="$3"
  local expect_num=""
  local jobCompleted=""

  expect_num=$(expect_num "$benchmark" "$benchmark_ns")
  jobCompleted=$(kubectl get benchmark "$benchmark" -n "$benchmark_ns" -ojson | jq -r .status.jobCompleted)
  echo "Wait for $expect_num  $benchmark jobs to be completed, sleep $sleep_time"

  while [ "$jobCompleted" != "$expect_num/$expect_num" ]; do
    sleep "$sleep_time"
    echo "Wait for $benchmark to be completed... $jobCompleted, sleep $sleep_time"
    jobCompleted=$(kubectl get benchmark "$benchmark" -n "$benchmark_ns" -ojson | jq -r .status.jobCompleted)
  done
  echo "Benchmark job completed"
}

save_benchmark() {
  local benchmark="$1"
  local benchmark_ns="$2"
  shift 2
  kubectl get benchmark "$benchmark" -n "$benchmark_ns" -ojson >"$DATAPATH/$benchmark.json"
}

collect_idle() {
  local args=(
    "-o" "idle"
    "--interval" "1000"
  )
  if ! "$NATIVE"; then
    docker run --rm -v "$DATAPATH":/data --network=host "$ENTRYPOINT_IMG" query "${args[@]}"
  else
    python ../cmd/main.py query "${args[@]}" || true
  fi
}

collect_data() {
  local benchmark="$1"
  local benchmark_ns="$2"
  local sleep_time="$3"
  shift 3
  [[ "$benchmark" != "customBenchmark" ]] && {
    kubectl apply -f "$CPE_BENCHMARK_DIR"/"$benchmark".yaml
    wait_for_benchmark "$benchmark" "$benchmark_ns" "$sleep_time"
    save_benchmark "$benchmark" "$benchmark_ns"
    kubectl delete -f "$CPE_BENCHMARK_DIR"/"$benchmark".yaml
  }
  local args=(
    "-i" "$benchmark"
    "-o" "${benchmark}_kepler_query"
    "-s" "$PROM_SERVER"
  )
  if ! "$NATIVE"; then
    docker run --rm -v "$DATAPATH":/data --network=host "$ENTRYPOINT_IMG" query "${args[@]}" || true
  else
    python ../cmd/main.py query "${args[@]}" || true
  fi
}

deploy_prom_dependency() {
  kubectl apply -f "$DEPOYMENT_DIR"/prom-kepler-rbac.yaml
  kubectl apply -f "$DEPOYMENT_DIR"/prom-np.yaml
}

train_model() {
  local query_response="$1"
  local pipeline_name="${PIPELINE_PREFIX}$2"
  echo "input=$query_response"
  echo "pipeline=$pipeline_name"
  local args=(
    "-i" "$query_response"
    "-p" "$pipeline_name"
    "--energy-source" "$ENERGY_SOURCE"
  )
  if ! "$NATIVE"; then
    echo "Train with docker"
    docker run --rm -v "$DATAPATH":/data "$ENTRYPOINT_IMG" train "${args[@]}" || true
  else
    echo "Train natively"
    python ../cmd/main.py train "${args[@]}" || true
  fi
}

prepare_cluster() {
  cluster_up
  deploy_kepler
  deploy_prom_dependency
  watch_service 9090 "monitoring" prometheus-k8s &
  reload_prometheus
}

watch_service() {
  local port="$1"
  local ns="$2"
  local svn="$3"
  shift 3
  kubectl port-forward --address localhost -n "$ns" service/"$svn" "$port":"$port"
}

collect() {
  collect_idle
  collect_data stressng cpe-operator-system 60
}

custom_collect() {
  collect_data customBenchmark
}

quick_collect() {
  collect_data sample cpe-operator-system 10
}

train() {
  train_model stressng_kepler_query "${VERSION}_stressng"
}

quick_train() {
  train_model sample_kepler_query "${VERSION}_sample"
}

custom_train() {
  train_model customBenchmark_kepler_query "${VERSION}_customBenchmark"
}

validate() {
  local benchmark="$1"
  local args=(
    "-i" "${benchmark}_kepler_query"
    "--benchmark" "$benchmark"
  )
  if ! "$NATIVE"; then
    docker run --rm -v "$DATAPATH":/data "$ENTRYPOINT_IMG" validate "${args[@]}"
  else
    python ../cmd/main.py validate "${args[@]}" || true
  fi
}

_export() {
  [[ $# -lt 4 ]] && {
    echo "need arguements: [machine_id] [path_to_models] [publisher] [benchmark_name]"
    return 1
  }
  local id="$1"
  local output="$2"
  local publisher="$3"
  local main_collect_input="$4"
  local include_raw="$5"
  shift 5

  local pipeline_name="${PIPELINE_PREFIX}${VERSION}_${main_collect_input}"
  local validate_input="${main_collect_input}_kepler_query"

  local args=(
    "--id" "$id"
    "-p" "$pipeline_name"
    "-i" "$validate_input"
    "--benchmark" "$main_collect_input"
    "--version" "$VERSION"
    "--publisher" "$publisher"
    "$include_raw"
  )
  echo "${args[@]}"
  if ! "$NATIVE"; then
    docker run --rm -v "$DATAPATH":/data -v "$output":/models "$ENTRYPOINT_IMG" export "${args[@]}" -o /models
  else
    python ../cmd/main.py export "${args[@]}" -o "$output"
  fi
  return 0
}

export_models() {
  local id="$1"
  local path_to_models="$2"
  local publisher="$3"
  local benchmark_name="$4"
  shift 4
  _export "$id" "$path_to_models" "$publisher" "$benchmark_name" || return 1
  return 0
}

export_models_with_raw() {
  local id="$1"
  local path_to_models="$2"
  local publisher="$3"
  local benchmark_name="$4"
  shift 4
  _export "$id" "$path_to_models" "$publisher" "$benchmark_name" "--include-raw true" || return 1
  return 0
}

cleanup() {
  clean_cpe_cr
  clean_deployment || true
  cluster_down
}

main() {
  local op="$1"
  shift 1
  mkdir -p "$DATAPATH"
  export MODEL_PATH=$DATAPATH
  export DATAPATH
  export PROMETHEUS_ENABLE
  export TEKTON_ENABLE
  export KIND_CLUSTER_NAME
  $op "$@"

}

main "$@"
