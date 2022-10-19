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
|KerasCompFullPipeline|Single-layer Linear Regression with Adam optimizer (learning_rate=0.5), loss='mae'|Full|:heavy_check_mark:

##### AbsComponentModelWeight (core, dram)
|Pipeline Name|Learning Approach|Features|Online|
|---|---|---|---|
|KerasCompWeightFullPipeline|Single-layer Linear Regression with Adam optimizer (learning_rate=0.5), loss='mae'|Full|:heavy_check_mark:

##### DynPower 
|Pipeline Name|Learning Approach|Features|Online|
|---|---|---|---|
ScikitMixed|Gradient Boosting Regressor|Full, Cgroup|x|

##### DynModelWeight (WIP)

##### DynComponentPower (WIP)

##### DynComponentModelWeight (WIP)

Example outputs:

*Weights*
```json
{"All_Weights": 
  {
  "Bias_Weight": 1.0, 
  "Categorical_Variables": {"cpu_architecture": {"Sky Lake": {"weight": 1.0}}}, 
  "Numerical_Variables": {"cpu_cycles": {"mean": 0, "variance": 1.0, "weight": 1.0}}
  }
}
```
*Archived model*
```bash
├── AbsComponentPower
│   ├── KerasCompFullPipeline.json
│   ├── core
│   │   ├── assets
│   │   ├── keras_metadata.pb
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── core_transform.pkl
│   ├── dram
│   │   ├── assets
│   │   ├── keras_metadata.pb
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   ├── dram_transform.pkl
│   └── metadata.json
└── DynPower
    ├── metadata.json
    ├── model.sav
    └── scaler.pkl
```


### Posting Model Weights [WIP]
```
/metrics
GET
```

## Power Models

There are two broad kinds of power models provided by the model server: absolute power model (Abs) and dynamic power model (Dyn). 

**Absolute Power Model (Abs):** power model trained by measured power (including the idle power)

**Dynamic Power Model (Dyn):** power model trained by delta power (measured power with workload - idle power)

The absolute power model is supposed to use for estimating node-level power while the dynamic power model is supposed to use for estimating pod-level power. 

Note that, the node-level power would be applied in the case that the measured power or low-level highly-dependent features (such as temperature sensors) are not available such as in virtualized systems, non-instrumented systems.

For each kind, the model can be in four following forms.

Key Name| Model Form|Output per single input|
---|---|---|
Power|achived model|single power value|
ModelWeight|weights for normalized linear regression model|single power value|
ComponentPower|achived model|multiple power values for each components (e.g., core, dram)
ComponentWeight|weights for normalized linear regression model|multiple power values for each components (e.g., core, dram)|

The achived model allows more choices of learning approachs to estimate the power which can expect higher accuracy and overfitting resolutions while weights of linear regression model provides simplicity and lightweight. **Accordingly, the achived model is suitable for the systems that run intensive and non-deterministic workloads which require accurate prediction while the weights of linear regression is suitable for the systems which run for deterministic workloads in general.**

Single power value can blend the inter-dependency between each power-consuming components into the model while multiple power values for each components can give more insight analysis with a precision tradeoff. **Accordingly, if we want insight analysis and have enough component-level training data to build the component model with acceptable accuracy, the multiple power values are preferable. Otherwise, single power value should be a common choice**


### Contributing
Please check the roadmap and guidelines to join us [here](./contributing.md).
