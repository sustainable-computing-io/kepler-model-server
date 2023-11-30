# Kepler power model training with Tekton
<!-- TOC tocDepth:2..3 chapterDepth:2..6 -->

- [Pre-requisite](#pre-requisite)
- [Deploy Tekton tasks and pipelines](#deploy-tekton-tasks-and-pipelines)
- [Run Tekton pipeline](#run-tekton-pipeline)
    - [Single train run](#single-train-run)
    - [Original complete run](#original-complete-run)

<!-- /TOC -->
## Pre-requisite
1. Cluster with Tekton
2. Prepare PersistentVolumeClaim `task-pvc` for workspace
    
    For simple hostpath,
    ```
    kubectl apply -f pvc/hostpath.yaml
    ```

    > The query, preprocess data, and models will be mounted to the hostpath: `/mnt`

## Deploy Tekton tasks and pipelines

```
kubectl apply -f tasks
kubectl apply -f pipelines
```

## Run Tekton pipeline
### Single train run
A single flow to apply a set of trainers to specific feature group and energy source.

![](../fig/tekton-single-train.png)

check [single-train](./pipelines/single-train.yaml) pipeline.

Example for AbsPower model:
    
```
kubectl apply -f abs-train-pipelinerun.yaml
```

Example of DynPower model:

```
kubectl apply -f dyn-train-pipelinerun.yaml
```

To customize feature metrics, set

parameters|value
---|---
THIRDPARTY_METRICS|customized metric list (use comma as delimiter)
FEATURE_GROUP|`ThirdParty`

To customize stressng workload, set
parameters|value
---|---
STRESS_BREAK_INTERVAL|break interval between each stress load
STRESS_TIMEOUT|stress duration (timeout to stop stress)
STRESS_ARGS|array of arguments for CPU frequency and stressng workload<br>- `CPU_FREQUENCY;STRESS_LOAD;STRESS_INSTANCE_NUM;STRESS_EXTRA_PARAM_KEYS;STRESS_EXTRA_PARAM_VALS`<br>* use `none` if not applicable for `CPU_FREQUENCY`, `STRESS_EXTRA_PARAM_KEYS`, and `STRESS_EXTRA_PARAM_VALS`

To customize preprocessing and training components
parameters|value
---|---
PIPELINE_NAME|pipeline name (output prefix/folder)
EXTRACTOR|extractor class (default or smooth)
ISOLATOR|isolator class (none, min, profile, or trainer)<br> For trainer isolator, ABS_PIPELINE_NAME must be set to use existing trained pipeline to estimate background power.
TRAINERS|list of trainer classes (use comma as delimiter)

### Original complete run
Apply a set of trainers to all available feature groups and energy sources

![](../fig/tekton-complete-train.png)

check [complete-train](./pipelines/complete-train.yaml) pipeline.

```
kubectl apply -f complete-pipelinerun.yaml
```

To customize `ThirdParty` feature group, set
parameters|value
---|---
THIRDPARTY_METRICS|customized metric list (use comma as delimiter)

Stressng load can be set similarly to [single train run](#single-train-run).

To customize pipeline components, `PIPELINE_NAME`, `EXTRACTOR`, `ISOLATOR`, and `ABS_PIPELINE_NAME` can be set similarly to [single train run](#single-train-run).

Instead of `TRAINERS`, original pipeline run use `ABS_TRAINERS` and `DYN_TRAINERS` to specify the list for trainers for AbsPower training and DynPower training respectively.