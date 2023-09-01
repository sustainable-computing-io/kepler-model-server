# Contributing
[Get started with Kepler Model Server.](https://sustainable-computing.io/kepler_model_server/get_started/)

- The main source codes are in [src directory](./src/).

## Improve components in training pipelines
Learn more details about [Training Pipeline](https://sustainable-computing.io/kepler_model_server/pipeline/)

### Introduce new feature group
- Define new feature group name `FeatureGroup` and update metric list map `FeatureGroups` in [train types](./src/util/train_types.py)

### Introduce new energy sources
- Define new energy source map `PowerSourceMap` in [train types](./src/util/train_types.py)

### Improve preprocessing method
- [extractor](./src/train/extractor/): convert from numerically aggregated metrics to per-second value
- [isolator](./src/train/isolator/): isolate background (idle) power from the collected power

### Introduce new learning method
- [trainer](./src/train/trainer/): apply learning method to build a model using extracted data and isolated data

## Model training
Learn more details about [model training](./model_training/)

### Introduce new benchmarks
The new benchmark must be supported by [CPE operator](https://github.com/IBM/cpe-operator) for automation.
Find [examples](https://github.com/IBM/cpe-operator/tree/main/examples).

`Benchmark` CR has a dependency on `BenchmarkOperator`. Default `BechmarkOperator` is to support [batch/v1/Job API](https://github.com/IBM/cpe-operator/blob/main/examples/none/cpe_v1_none_operator.yaml).

### Add new trained models
TBD

## Source improvement
Any improvement in `src` and `cmd`.

## Test and CI improvement
Any improvement in `tests`, `dockerfiles`, `manifests` and `.github/workflows`
