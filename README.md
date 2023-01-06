# Kepler Model Server

### Getting Power Models
```
/model
POST
```

Parameters
|key|value|description
|---|---|---|
|metrics|list of string|list of available input features (measured metrics)*
|output_type|either of the following values: *AbsPower*, *AbsModelWeight*, *AbsComponentPower*, *AbsComponentModelWeight*, *DynPower*,  *DynModelWeight*, *DynComponentPower*, *DynComponentModelWeight*|the requested model kinds and forms \**
model_name (optional)|string|fixed pipeline name* (auto-select the model with lowest error if not specified)
filters (optional)|string|expression in the form *attribute1*:*threshold1*; *attribute2*:*threshold2*

\* refer to pipeline and features definitions in [here](./doc/train_pipeline.md)

\** refer to power model choices described [here](#power-models)

#### Available model training pipelines
##### AbsPower
|Pipeline Name|Learning Approach|Features|Online|
|---|---|---|---|
|KerasFullPipeline|Single-layer Linear Regression with Adam optimizer (learning_rate=0.5), loss='mae'|Full|:heavy_check_mark:

##### AbsModelWeight (WIP)

##### AbsComponentPower (core,dram)
|Pipeline Name|Learning Approach|Features|Online|
|---|---|---|---|
|KerasCompFullPipeline|Single-layer Linear Regression with Adam optimizer (learning_rate=0.01), loss='mae'|Full|:heavy_check_mark:

##### AbsComponentModelWeight (core, dram)
|Pipeline Name|Learning Approach|Features|Online|
|---|---|---|---|
|KerasCompWeightFullPipeline|Single-layer Linear Regression with Adam optimizer (learning_rate=0.01), loss='mae'|Full|:heavy_check_mark:

##### DynPower 
|Pipeline Name|Learning Approach|Features|Online|
|---|---|---|---|
ScikitMixed|Gradient Boosting Regressor|Full, Cgroup|x|


### Posting Model Weights [WIP]
```
/metrics
GET
```

### Contributing
Please check the roadmap and guidelines to join us [here](./contributing.md).
