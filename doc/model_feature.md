# Model Input Features
The energy-related metrics collected and exported by [Kepler](https://github.com/sustainable-computing-io/kepler) are groupped by the input source as below.

**Kepler metric:**
Kepler metrics|Purpose of use
---|---
kepler_node_platform_joules_total|labeled values of total Abs/Dyn model
kepler_node_<power_component>_joules_total|labeled values of component Abs/Dyn model
kepler_node_energy_stat|features for Abs model
kepler_node_cpu_scaling_frequency_hertz|cpu_frequency feature for Abs model
kepler_pod_energy_stat|features for Dyn model
kepler_container_cpu_cpu_time_us|compute cpu_frequency feature for Dyn model

**Feature group:**
Group Name|Features
---|---
Full|WORKLOAD_FEATURES + SYSTEM_FEATURES
WorkloadOnly|WORKLOAD_FEATURES
CounterOnly|COUNTER_FEAUTRES
CgroupOnly|CGROUP_FEATURES
BPFOnly|BPF_FEATURES
KubeletOnly|KUBELET_FEATURES

*Note:* WORKLOAD_FEATURES includes COUNTER_FEAUTRES, CGROUP_FEATURES, IO_FEATURES, BPF_FEATURES and KUBELET_FEATURES.


**Feature list:**
Features|Kepler exporting metric labels in `kepler_pod_energy_stat` metric
---|---
COUNTER_FEAUTRES| - cache_miss<br>- cpu_cycles<br>- cpu_instr
CGROUP_FEATURES| - cgroupfs_cpu_usage_us<br>- cgroupfs_memory_usage_bytes<br>- cgroupfs_system_cpu_usage_us<br>- cgroupfs_user_cpu_usage_us
IO_FEATURES|- bytes_read<br>- bytes_writes
BPF_FEATURES| - cpu_time
KUBELET_FEATURES|- container_cpu_usage_seconds_total<br>- container_memory_working_set_bytes
SYSTEM_FEATURES| - cpu_architecture <br> - cpu_frequency (TBD)

* Currently, the cpu_frequency per pod is computed with combination of `kepler_pod_energy_stat` and `kepler_container_cpu_cpu_time_us`
  
Also see [metric label declaration](https://github.com/sustainable-computing-io/kepler/blob/c8f0528980944d3952f001ac09cd4e285cebe73e/pkg/collector/prometheus_collector.go#L40-L80)

