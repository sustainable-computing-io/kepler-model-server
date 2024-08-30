# Contribute to power profiling and model training

<!--toc:start-->

- [Contribute to power profiling and model training](#contribute-to-power-profiling-and-model-training)
  - [Requirements](#requirements)
  - [Pre-step](#pre-step)
  - [Setup](#setup)
    - [Prepare cluster](#prepare-cluster)
    - [From scratch (no target kubernetes cluster)](#from-scratch-no-target-kubernetes-cluster)
    - [For managed cluster](#for-managed-cluster)
    - [Run benchmark and collect metrics](#run-benchmark-and-collect-metrics)
    - [With manual execution](#with-manual-execution)
  - [Clean up](#clean-up)

<!--toc:end-->

## Requirements

- git > 2.22
- kubectl
- yq, jq
- power meter is available

## Pre-step

- Fork and clone this repository and move to `model_training` folder

```bash
git clone
cd model_training
```

## Setup

### Prepare cluster

### From scratch (no target kubernetes cluster)

> Note: port 9090 and 5101 should not being used. It will be used in port-forward for prometheus and kind registry respectively

  ```bash
  ./script.sh prepare_cluster
  ```

The script will:

- create a kind cluster `kind-for-training` with registry at port `5101`.
- deploy Prometheus.
- deploy Prometheus RBAC and node port to `30090` port on kind node which will be forwarded to `9090` port on the host.
- deploy service monitor for kepler and reload to Prometheus server

### For managed cluster

Please confirm the following requirements:

- Kepler installation
- Prometheus installation
- Kepler metrics are exported to Promtheus server
- Prometheus server is available at `http://localhost:9090`. Otherwise, set environment `PROM_SERVER`.

### Run benchmark and collect metrics

There are two options to run the benchmark and collect the metrics, [CPE-operator](https://github.com/IBM/cpe-operator) with manual script and [Tekton Pipeline](https://github.com/tektoncd/pipeline).

> The adoption of the CPE operator is slated for deprecation. We are on transitioning to the automation of collection and training processes through the Tekton pipeline. Nevertheless, the CPE operator might still be considered for usage in customized benchmarks requiring performance values per sub-workload within the benchmark suite.

- [Tekton Pipeline Instruction](./tekton/README.md)

- [CPE Operator Instruction](./cpe_script_instruction.md)

### With manual execution

In addition to the above two automation approach, you can manually run your own benchmarks, then collect, train, and export the models by the entrypoint `cmd/main.py`

[Manual Metric Collection and Training with Entrypoint](./cmd_instruction.md)

## Clean up

For kind-for-training cluster:

```bash
./script.sh cleanup
```
