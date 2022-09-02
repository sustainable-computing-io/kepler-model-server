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

COUNTER_FEAUTRES = ["curr_cache_miss", "curr_cpu_cycles", "curr_cpu_instr"]
CGROUP_FEATURES = ["curr_cgroupfs_cpu_usage_us", "curr_cgroupfs_memory_usage_bytes", "curr_cgroupfs_system_cpu_usage_us", "curr_cgroupfs_user_cpu_usage_us"]
IO_FEATURES = ["curr_bytes_read", "curr_bytes_writes"]
BPF_FEATURES = ["curr_cpu_time"]
KUBELET_FEATURES =['container_cpu_usage_seconds_total', 'container_memory_working_set_bytes']
WORKLOAD_FEATURES = COUNTER_FEAUTRES + CGROUP_FEATURES + IO_FEATURES + BPF_FEATURES + KUBELET_FEATURES

CATEGORICAL_LABEL_TO_VOCAB = {
                    "cpu_architecture": ["Sandy Bridge", "Ivy Bridge", "Haswell", "Broadwell", "Sky Lake", "Cascade Lake", "Coffee Lake", "Alder Lake"] 
                    }

NODE_STAT_POWER_LABEL = ["node_curr_energy_in_pkg_joule", "node_curr_energy_in_core_joule", "node_curr_energy_in_dram_joule", "node_curr_energy_in_uncore_joule", "node_curr_energy_in_gpu_joule", "node_curr_energy_in_other_joule"]

class FeatureGroup(enum.Enum):
   Full = 1
   WorkloadOnly = 2
   CounterOnly = 3
   CgroupOnly = 4
   BPFOnly = 5
   KubeletOnly = 6
   Unknown = 99

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

if __name__ == '__main__':
    for g, g_features in FeatureGroups.items():
        shuffled_features = g_features.copy()
        random.shuffle(shuffled_features)
        get_group = get_feature_group(shuffled_features)
        assert get_group == g, "must be " + str(g)
    assert get_feature_group([]) == FeatureGroup.Unknown, "must be unknown"
