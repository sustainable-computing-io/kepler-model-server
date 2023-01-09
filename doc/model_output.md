## Model Output Type

There are two broad kinds of output types of power models provided by the model server: absolute power model (Abs) and dynamic power model (Dyn). 

**Absolute Power Model (Abs):** power model trained by measured power (including the idle power)

**Dynamic Power Model (Dyn):** power model trained by delta power (measured power with workload - idle power)

The absolute power model is supposed to use for estimating node-level power while the dynamic power model is supposed to use for estimating pod-level power. 

Note that, the node-level power would be applied in the case that the measured power or low-level highly-dependent features (such as temperature sensors) are not available such as in virtualized systems, non-instrumented systems.

For each kind, the model can be in four following forms.

Key Name| Model Form|Output per single input|
---|---|---|
Power|[archived model](./model_format.md#2-archived-model)|single power value|
ModelWeight|[weights for normalized linear regression model](./model_format.md#1-model-weight-for-linear-regression)|single power value|
ComponentPower|[archived model](./model_format.md#2-archived-model)|multiple power values for each components (e.g., core, dram)
ComponentModelWeight|[weights for normalized linear regression model](./model_format.md#1-model-weight-for-linear-regression)|multiple power values for each components (e.g., core, dram)|

The archived model allows more choices of learning approachs to estimate the power which can expect higher accuracy and overfitting resolutions while weights of linear regression model provides simplicity and lightweight. **Accordingly, the archived model is suitable for the systems that run intensive and non-deterministic workloads which require accurate prediction while the weights of linear regression is suitable for the systems which run for deterministic workloads in general.**

Single power value can blend the inter-dependency between each power-consuming components into the model while multiple power values for each components can give more insight analysis with a precision tradeoff. **Accordingly, if we want insight analysis and have enough component-level training data to build the component model with acceptable accuracy, the multiple power values are preferable. Otherwise, single power value should be a common choice**

Accordingly, there are 8 output types declared as below. 
```python
class ModelOutputType(enum.Enum):
    AbsPower = 1
    AbsModelWeight = 2
    AbsComponentPower = 3
    AbsComponentModelWeight = 4
    DynPower = 5
    DynModelWeight = 6
    DynComponentPower = 7
    DynComponentModelWeight = 8
```