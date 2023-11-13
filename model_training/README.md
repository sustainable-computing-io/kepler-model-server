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

## 3. Collect metrics without running benchmark
Users might want to run a custom benchmark outside of `kepler-model-server` and collect metrics for training the model using `kepler-model-server`.

### Custom Benchmark

    ./script.sh custom_collect

Use either `interval` or [`start_time`, `end_time`] options to set the desired time window for metrics collection from Prometheus.

## 4. Validate collected metrics
Validation of metrics happens by default at the time of their collection. It is also possible to validate the collected metrics explicitly.

### Quick sample

    ./script.sh validate sample

### Full run

    ./script.sh validate stressng

### Custom benchmark

    ./script.sh validate customBenchmark

## 5. Profile and train

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

#### Custom benchmark

```
./script.sh custom_train
```

Training output will be in `/data` folder by default. The folder contains:
- preprocessed data from Prometheus query response
- profiles
- models in a pipeline heirachy 

### < native with python environment >
Compatible version: python 3.8

- Prepare environment:

    ```bash
    pip install -r ../dockerfiles/requirements.txt
    ```

- Run

    ```bash
    NATIVE="true" ./script.sh train
    ```

## Clean up

### For kind-for-training cluster

Run
```
./script.sh cleanup
```

## Upload model to repository

1. Fork `kepler-model-db`.

1. Validate and make a copy by export command. Need to define `machine id`, `local path to forked kepler-model-db/models`, `author github account` and `benchmark type`.

    Run
    ```
    ./script.sh export <machine id> <path to kepler-model-db/models> <author github account> <benchmark type>
    ```

    If you also agree to share the raw data (preprocessed data and archived file of full pipeline), run

    ```
    ./script.sh export_with_raw <machine id> <path to kepler-model-db/models> <author github account> <benchmark type>
    ```

    - set `NATIVE="true"` to export natively.
    - Benchmark type accepts one of the values `sample`, `stressng` or `customBenchmark`.

2. Add information of your machine in `./models/README.md` in `kepler-model-db`. You may omit any column as needed.
3. Push PR to `kepler-model-db.
