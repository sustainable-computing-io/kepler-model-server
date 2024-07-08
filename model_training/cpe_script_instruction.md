# Kepler power model training wih CPE operator

Previous Step: [Prepare cluster](./README.md#1-prepare-cluster)

## 1. Deploy CPE operator
    
- For Kind cluster provided by `script.sh`:

        ./script.sh deploy_cpe_operator

- For managed cluster:

    ```bash
    CLUSTER_PROVIDER= # kind|k8s|ocp 
    kubectl apply -f https://raw.githubusercontent.com/IBM/cpe-operator/main/examples/deployment/${CLUSTER_PROVIDER}-deploy.yaml
    ```

## 2. Run Workload

### 2.1. For managed cluster, make sure that cluster is in an idle state with only control plane workload

### 2.2. Run workload

There are three available mode: `quick` for testing, `full` for pre-defined stressng, and `custom` for non-CPE benchmarks.

### (1) Quick sample

    ./script.sh quick_collect

This is only for testing purpose.

### (2) Stressng (standard workload)

    ./script.sh collect

It might take an hour to run and collect all benchmarks. Output including CPE CR and Prometheus query response will be in `data` folder by default.

### (3) Custom Benchmark
With CPE operator, the start and the end time of each pod will be recorded. However, users might want to run a custom benchmark outside of `kepler-model-server` and collect metrics for training the model using `kepler-model-server`. In that case, user can define either `interval` or [`start_time`, `end_time`] options to set the desired time window for metrics collection from Prometheus.

    ./script.sh custom_collect

## 3. Validate collected metrics
Validation of metrics happens by default at the time of their collection. It is also possible to validate the collected metrics explicitly.

### Quick sample

    ./script.sh validate sample

### Full run

    ./script.sh validate stressng

### Custom benchmark

    ./script.sh validate customBenchmark

## 4. Profile and train

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
Compatible version: python 3.10

- Install [`hatch`](https://hatch.pypa.io/latest/install/)
- Prepare environment:

    ```bash
    hatch shell
    ```

- Run

    ```bash
    NATIVE="true" ./script.sh train
    ```

## Upload model to repository

1. Fork `kepler-model-db`.

1. Validate and make a copy by export_models command. Need to define `machine id`, `local path to forked kepler-model-db/models`, `author github account` and `benchmark type`.

    Run
    ```
    ./script.sh export_models <machine id> <path to kepler-model-db/models> <author github account> <benchmark type>
    ```

    If you also agree to share the raw data (preprocessed data and archived file of full pipeline), run

    ```
    ./script.sh export_models_with_raw <machine id> <path to kepler-model-db/models> <author github account> <benchmark type>
    ```

    - set `NATIVE="true"` to export natively.
    - Benchmark type accepts one of the values `sample`, `stressng` or `customBenchmark`.

2. Add information of your machine in `./models/README.md` in `kepler-model-db`. You may omit any column as needed.
3. Push PR to `kepler-model-db.
