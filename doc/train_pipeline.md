# Trainning Pipeline
The model server contains a collection of trainning pipelines in a unified format defined in `server/train/pipeline.py`.

Each pipeline can be categorized into one of the group defined in `/server/train/train_types.py` according to its train features.

Also check [model input features](./model_feature.md).

---

### Guideline to define a new pipeline class
refer to [pipeline.py](../server/train/pipeline.py)
1. Define the following attributes.

    Attribute|Description
    ---|---
    *model_name*| unique ID inside the feature group
    *model_class* | class of modeling approach, currently support only `keras` and `scikit` for Keras Model and Scikit-learn Regressor, respectively.
    *model_file*| model filename, this will be load accoring to *model_class* (e.g., `model.h5` for `keras`, `model.sav` for `scikit`)
    *features*| list of train features referring to kepler exporting labels in `pod_energy_stat` metric


2. implement `train(self, pod_stat_data, node_stat_data, freq_data, pkg_data)`

    Argument|Prometheus Metric|Description
    ---|---|---
    *pod_stat_data*| `pod_energy_stat` |resource usage per pod
    *node_stat_data* | `node_energy_stat`|resource usage and measured energy per node
    *freq_data*| `node_cpu_scaling_frequency_hertz`|CPU frequency per core per node
    *pkg_data*| `node_package_energy_millijoule`|measured energy per package per node

    expected actions in this function:
    - train a prediction model from the given data
    - update the trained model and its metadata if improved
      - metadata attributes must be compatible with [kepler-estimator](https://github.com/sustainable-computing-io/kepler-estimator/blob/d351cbfadd67f5f270f223c2e7ebeb225f06a070/src/model/load.py#L17))
      - input of the model will be array in a shape (m, n) where m = number of pod, n = number of model features
      - output of the model must a single power per pod

Check the output structure [here](./model_format.md)