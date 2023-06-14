# Kepler Power Model
This repository contains source code related to Kepler power model. The modules in this reposioty connects to [core Kepler project](https://github.com/sustainable-computing-io/kepler) and [kepler-model-db](https://github.com/sustainable-computing-io/kepler-model-db) as below.
![](./doc/fig/model-server-components-simplified.png)

## Usage
### Getting Powers from Estimator
**module:** estimator (src/estimate/estimator.py)
```
/tmp/estimator.socket
```
Parameters
|key|value|description
|---|---|---|
|metrics|list of string|list of available input features (measured metrics)
|output_type|either of the following values: *AbsPower* (for node-level power model), *DynPower* (for container-level power model)|the requested model type 
|trainer_name (optional)|string|filter model with trainer name.
|filter (optional)|string|expression in the form *attribute1*:*threshold1*; *attribute2*:*threshold2*.

### Getting Power Models from Model Server 
**module:** server (src/server/model_server.py)
```
:8100/model
POST
```

Parameters
|key|value|description
|---|---|---|
|metrics|list of string|list of available input features (measured metrics)
|output_type|either of the following values: *AbsPower* (for node-level power model), *DynPower* (for container-level power model)|the requested model type 
|weight|boolean|return model weights in json format if true. Otherwise, return model in zip file format.
|trainer_name (optional)|string|filter model with trainer name.
|node_type (optional)|string|filter model with node type.
|filter (optional)|string|expression in the form *attribute1*:*threshold1*; *attribute2*:*threshold2*.

### Posting Model Weights [WIP]
**module:** server (src/server/model_server.py)
```
/metrics
GET
```

### Online Trainer [WIP]
**module:** online trainer (src/train/online_trainer.py)
running as a sidecar to server
```
periodically query prometheus metric server on SAMPLING INTERVAL
```

### Profiler [WIP]
**module:** profiler (src/profile/profiler.py)


### Offline Trainer [WIP]
**module:** offline trainer (src/train/offline_trainer.py)

## Test
Build image for testing, run 
```
make build-test
```

|Test case|Command|
|---|---|
|[Training pipeline](./tests/README.md#pipeline)|make test-pipeline|
|[Model server](./tests/README.md#estimator-model-request-to-model-server)|make test-model-server|
|[Estimator](./tests/README.md#estimator-power-request-from-collector)|make test-estimator|

For more test information, check [here](./tests/).

### Contributing
Please check the roadmap and guidelines to join us [here](./contributing.md).
