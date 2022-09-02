# Trainning Pipeline
The model server contains a collection of trainning pipelines in a unified format defined in `server/train/pipeline.py`.

Each pipeline can be categorized into one of the group defined in `/server/train/train_types.py` according to its train features.


Group Name|Features
---|---
Full|WORKLOAD_FEATURES + SYSTEM_FEATURES
WorkloadOnly|WORKLOAD_FEATURES
CounterOnly|COUNTER_FEAUTRES
CgroupOnly|CGROUP_FEATURES + IO_FEATURES
BPFOnly|BPF_FEATURES
KubeletOnly|KUBELET_FEATURES



*Note:* WORKLOAD_FEATURES includes COUNTER_FEAUTRES, CGROUP_FEATURES + IO_FEATURES, BPF_FEATURES and KUBELET_FEATURES.


Features|Kepler exporting labels in `pod_energy_stat` metric
---|---
COUNTER_FEAUTRES| - curr_cache_miss<br>- curr_cpu_cycles<br>- curr_cpu_instr
CGROUP_FEATURES+IO_FEATURES| - curr_cgroupfs_cpu_usage_us<br>- curr_cgroupfs_memory_usage_byte<br>- curr_cgroupfs_system_cpu_usage_us<br>- curr_cgroupfs_user_cpu_usage_us<br>-curr_bytes_read<br>-curr_bytes_writes
BPF_FEATURES| - curr_cpu_time
KUBELET_FEATURES|- container_cpu_usage_seconds_total<br>- container_memory_working_set_bytes
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

