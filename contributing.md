# Contributing

[Get started with Kepler Model Server.](https://sustainable-computing.io/kepler_model_server/get_started/)

- The main source codes are in [src directory](./src/).

## PR Hands-on

- Create related [issue](https://github.com/sustainable-computing-io/kepler-model-server/issues) with your name assigned first (if not exist).

- Set required secret and environment for local repository test if needed. Check below table.

| Objective | Required Secret | Required Environment |
| --------- | --------------- |----------------------|
| Push to private repo |BOT_NAME, BOT_TOKEN | IMAGE_REPO |
| Change on base image | BOT_NAME, BOT_TOKEN | IMAGE_REPO |
| Save data/models to AWS COS | AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY,AWS_REGION | |

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

### Tekton

Create workload `Task` and provide example `Pipeline` to run.

### Add new trained models

TBD

## Source improvement

Any improvement in `src` and `cmd`.

## Test and CI improvement

Any improvement in `tests`, `dockerfiles`, `manifests` and `.github/workflows`

## Documentation

Detailed documentation should be posted to [kepler-doc](https://github.com/sustainable-computing-io/kepler-doc) repository.
