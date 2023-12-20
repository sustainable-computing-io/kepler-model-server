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
3. deploy Prometheus RBAC and node port to `30090` port on kind node which will be forwarded to `9090` port on the host.
4. deploy service monitor for kepler and reload to Prometheus server

### For managed cluster

Please confirm the following requirements:
1. Kepler installation
2. Prometheus installation
3. Kepler metrics are exported to Promtheus server
4. Prometheus server is available at `http://localhost:9090`. Otherwise, set environment `PROM_SERVER`.

## 2. Run benchmark and collect metrics

There are two options to run the benchmark and collect the metrics, [CPE-operator](https://github.com/IBM/cpe-operator) with manual script and [Tekton Pipeline](https://github.com/tektoncd/pipeline). 

> The adoption of the CPE operator is slated for deprecation. We are on transitioning to the automation of collection and training processes through the Tekton pipeline. Nevertheless, the CPE operator might still be considered for usage in customized benchmarks requiring performance values per sub-workload within the benchmark suite.

### [Tekton Pipeline Instruction](./tekton/README.md)

### [CPE Operator Instruction](./cpe_script_instruction.md)

## Clean up

### For kind-for-training cluster

Run
```
./script.sh cleanup
```
