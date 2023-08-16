# Contribute to power profiling amd model training

## Requirements
- git > 2.22
- kubectl
- yq, jq
- power meter is available

## Pre-step
1. Fork and clone this repository and move to profile folder
    ```bash
    git clone
    cd model_training
    chmod +x script.sh
    ```
## 1. Prepare cluster

### From scratch (no target kubernetes cluster)
- port 9090 and 5101 not being used (will be used in port-forward for prometheus and kind registry respectively)

Run
```
./script.sh prepare_cluster
```
The script will 
1. create a kind cluster `kind-for-training` with registry at port `5101`.
2. deploy Prometheus.
3. deploy CPE operator for automating the benchmarks.
4. deploy Prometheus RBAC and node port to `30090` port on kind node which will be forwarded to `9090` port on the host.
5. deploy service monitor for both CPE and kepler and reload to Prometheus server

### For managed cluster

Please confirm the following requirements:
1. Kepler installation
2. Prometheus installation
3. CPE >v1.0.0 installation
4. Kepler metrics are exported to Promtheus server
5. Prometheus server is available at `http://localhost:9090`. Otherwise, set environment `PROM_SERVER`.
   
## 2. Run benchmark and collect metric
### 2.1. For managed cluster, make sure that cluster is in an idle state with only control plane workload

### 2.2. Run
### Quick sample

    ./script.sh quick_collect

This is only for testing purpose.

### Full run

    ./script.sh collect


It might take an hour to run and collect all benchmarks. Output including CPE CR and Prometheus query response will be in `data` folder by default.

## 3. Profile and train 

You can train the model by using docker image which require no environment setup but can be limited by docker limitation or train the model natively by setting up your python environment as follows.

### < via docker image >

#### Quick sample

```
./script.sh quick_train
```


#### Full run 

```
./script.sh train
```

Training output will be in `/data` folder by default. The folder contains:
- preprocessed data from Prometheus query response
- profiles
- models in a pipeline heirachy 

### < native with python environment >

- Prepare environment:

    ```bash
    pip install -r ../dockerfiles/requirements.txt
    ```

- Run

    ```bash
    # set training parameter
    export CPE_DATAPATH=$(pwd)/data
    export QUERY_RESPONSE=coremark_kepler_query,stressng_kepler_query,parsec_kepler_query
    export VERSION=0.6
    export PIPELINE_NAME=$(uname)-$(uname -r)-$(uname -m)_v${VERSION}_train
    export ENERGY_SOURCE=rapl
    echo "input=$QUERY_RESPONSE"
    echo "pipeline=$PIPELINE_NAME"

    # train
    python ../cmd/main.py train -i ${QUERY_RESPONSE} -p ${PIPELINE_NAME} --profile idle --isolator profile --energy-source ${ENERGY_SOURCE}
    ```

## Clean up

### For kind-for-training cluster

Run
```
./script.sh cleanup
```

## Upload model to repository
TBD
