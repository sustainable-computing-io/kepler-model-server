###########################################################
## types.py
## 
## defines
## - collection of features
## - feature groups
## - power labels
##
###########################################################

import enum
import random

SYSTEM_FEATURES = ["nodeInfo", "cpu_scaling_frequency_hertz"]

COUNTER_FEAUTRES = ["cache_miss", "cpu_cycles", "cpu_instructions"]
CGROUP_FEATURES = ["cgroupfs_cpu_usage_us", "cgroupfs_memory_usage_bytes", "cgroupfs_system_cpu_usage_us", "cgroupfs_user_cpu_usage_us"]
BPF_FEATURES = ["bpf_cpu_time_us"]
IRQ_FEATURES = ["bpf_block_irq", "bpf_net_rx_irq", "bpf_net_tx_irq"]
KUBELET_FEATURES =['kubelet_memory_bytes', 'kubelet_cpu_usage']
WORKLOAD_FEATURES = COUNTER_FEAUTRES + CGROUP_FEATURES + BPF_FEATURES + IRQ_FEATURES + KUBELET_FEATURES

PowerSourceMap = {
    "rapl": ["package", "core", "uncore", "dram"],
    "acpi": ["platform"]
}

class FeatureGroup(enum.Enum):
   Full = 1
   WorkloadOnly = 2
   CounterOnly = 3
   CgroupOnly = 4
   BPFOnly = 5
   KubeletOnly = 6
   IRQOnly = 7
   CounterIRQCombined = 8
   Unknown = 99

class ModelOutputType(enum.Enum):
    AbsPower = 1
    DynPower = 2

def deep_sort(elements):
    sorted_elements = elements.copy()
    sorted_elements.sort()
    return sorted_elements

FeatureGroups = {
    FeatureGroup.Full: deep_sort(WORKLOAD_FEATURES + SYSTEM_FEATURES),
    FeatureGroup.WorkloadOnly: deep_sort(WORKLOAD_FEATURES),
    FeatureGroup.CounterOnly: deep_sort(COUNTER_FEAUTRES),
    FeatureGroup.CgroupOnly: deep_sort(CGROUP_FEATURES),
    FeatureGroup.BPFOnly: deep_sort(BPF_FEATURES),
    FeatureGroup.KubeletOnly: deep_sort(KUBELET_FEATURES),
    FeatureGroup.IRQOnly: deep_sort(IRQ_FEATURES),
    FeatureGroup.CounterIRQCombined: deep_sort(COUNTER_FEAUTRES + IRQ_FEATURES),
}

def get_feature_group(features):
    sorted_features = sort_features(features)
    for g, g_features in FeatureGroups.items():
        print(g_features, features)
        if sorted_features == g_features:
            return g
    return FeatureGroup.Unknown

def get_valid_feature_groups(features):
    valid_fgs = []
    for fg_key, fg_features in FeatureGroups.items():
        valid = True
        for f in fg_features:
            if f not in features:
                valid = False
                break
        if valid:
            valid_fgs += [fg_key]
    return valid_fgs

def is_weight_output(output_type):
    if output_type == ModelOutputType.AbsModelWeight:
        return True
    if output_type == ModelOutputType.AbsComponentModelWeight:
        return True
    if output_type == ModelOutputType.DynModelWeight:
        return True
    if output_type == ModelOutputType.DynComponentModelWeight:
        return True
    return False

if __name__ == '__main__':
    for g, g_features in FeatureGroups.items():
        shuffled_features = g_features.copy()
        random.shuffle(shuffled_features)
        get_group = get_feature_group(shuffled_features)
        assert get_group == g, "must be " + str(g)
    assert get_feature_group([]) == FeatureGroup.Unknown, "must be unknown"