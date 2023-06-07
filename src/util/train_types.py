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
# "rapl": ["package", "core", "uncore", "dram"],
    "rapl": ["package"],
    "acpi": ["platform"]
}

CATEGORICAL_LABEL_TO_VOCAB = {
                    "cpu_architecture": ["Sandy Bridge", "Ivy Bridge", "Haswell", "Broadwell", "Sky Lake", "Cascade Lake", "Coffee Lake", "Alder Lake"],
                    "nodeInfo": ["1"],
                    "cpu_scaling_frequency_hertz": ["1GHz", "2GHz", "3GHz"],
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
   Basic = 9
   Unknown = 99

class ModelOutputType(enum.Enum):
    AbsPower = 1
    DynPower = 2

def is_support_output_type(output_type_name):
    return any(output_type_name == item.name for item in ModelOutputType)

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
    FeatureGroup.Basic: deep_sort(COUNTER_FEAUTRES+CGROUP_FEATURES+KUBELET_FEATURES+BPF_FEATURES),
}

all_feature_groups = [fg.name for fg in FeatureGroups.keys()]

def get_feature_group(features):
    sorted_features = deep_sort(features)
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