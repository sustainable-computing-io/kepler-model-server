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

SYSTEM_FEATURES = ["cpu_architecture"]

COUNTER_FEAUTRES = ["cache_miss", "cpu_cycles", "cpu_instr"]
CGROUP_FEATURES = ["cgroupfs_cpu_usage_us", "cgroupfs_memory_usage_bytes", "cgroupfs_system_cpu_usage_us", "cgroupfs_user_cpu_usage_us"]
IO_FEATURES = ["bytes_read", "bytes_writes"]
BPF_FEATURES = ["cpu_time"]
KUBELET_FEATURES =['kubelet_memory_bytes', 'kubelet_cpu_usage']
WORKLOAD_FEATURES = COUNTER_FEAUTRES + CGROUP_FEATURES + IO_FEATURES + BPF_FEATURES + KUBELET_FEATURES

CATEGORICAL_LABEL_TO_VOCAB = {
                    "cpu_architecture": ["Sandy Bridge", "Ivy Bridge", "Haswell", "Broadwell", "Sky Lake", "Cascade Lake", "Coffee Lake", "Alder Lake"] 
                    }

NODE_STAT_POWER_LABEL = ["energy_in_pkg_joule", "energy_in_core_joule", "energy_in_dram_joule", "energy_in_uncore_joule", "energy_in_gpu_joule", "energy_in_other_joule"]

class FeatureGroup(enum.Enum):
   Full = 1
   WorkloadOnly = 2
   CounterOnly = 3
   CgroupOnly = 4
   BPFOnly = 5
   KubeletOnly = 6
   Unknown = 99

class ModelOutputType(enum.Enum):
    AbsPower = 1
    AbsModelWeight = 2
    AbsComponentPower = 3
    AbsComponentModelWeight = 4
    DynPower = 5
    DynModelWeight = 6
    DynComponentPower = 7
    DynComponentModelWeight = 8

CORE_COMPONENT = 'core'
DRAM_COMPONENT = 'dram'

POWER_COMPONENTS = [CORE_COMPONENT, DRAM_COMPONENT]

def sort_features(features):
    sorted_features = features.copy()
    sorted_features.sort()
    return sorted_features

FeatureGroups = {
    FeatureGroup.Full: sort_features(WORKLOAD_FEATURES + SYSTEM_FEATURES),
    FeatureGroup.WorkloadOnly: sort_features(WORKLOAD_FEATURES),
    FeatureGroup.CounterOnly: sort_features(COUNTER_FEAUTRES),
    FeatureGroup.CgroupOnly: sort_features(CGROUP_FEATURES + IO_FEATURES),
    FeatureGroup.BPFOnly: sort_features(BPF_FEATURES),
    FeatureGroup.KubeletOnly: sort_features(KUBELET_FEATURES)
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