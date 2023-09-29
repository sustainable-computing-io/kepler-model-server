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
from typing import List

SYSTEM_FEATURES = ["node_info", "cpu_scaling_frequency_hertz"]

COUNTER_FEAUTRES = ["cache_miss", "cpu_cycles", "cpu_instructions"]
CGROUP_FEATURES = ["cgroupfs_cpu_usage_us", "cgroupfs_memory_usage_bytes", "cgroupfs_system_cpu_usage_us", "cgroupfs_user_cpu_usage_us"]
BPF_FEATURES = ["bpf_cpu_time_us"]
IRQ_FEATURES = ["bpf_block_irq", "bpf_net_rx_irq", "bpf_net_tx_irq"]
KUBELET_FEATURES =['kubelet_memory_bytes', 'kubelet_cpu_usage']
ACCELERATE_FEATURES = ['accelerator_intel_qat']
WORKLOAD_FEATURES = COUNTER_FEAUTRES + CGROUP_FEATURES + BPF_FEATURES + IRQ_FEATURES + KUBELET_FEATURES + ACCELERATE_FEATURES
BASIC_FEATURES = COUNTER_FEAUTRES + CGROUP_FEATURES + BPF_FEATURES + KUBELET_FEATURES

PowerSourceMap = {
    "rapl": ["package", "core", "uncore", "dram"],
    "acpi": ["platform"]
}

PACKAGE_ENERGY_COMPONENT_LABEL = ["package"]
DRAM_ENERGY_COMPONENT_LABEL = ["dram"]
CORE_ENERGY_COMPONENT_LABEL = ["core"]

CATEGORICAL_LABEL_TO_VOCAB = {
                    "cpu_architecture": ["Sandy Bridge", "Ivy Bridge", "Haswell", "Broadwell", "Sky Lake", "Cascade Lake", "Coffee Lake", "Alder Lake"],
                    "node_info": ["1"],
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
    BPFIRQ = 10
    AcceleratorOnly = 11
    Unknown = 99

class EnergyComponentLabelGroup(enum.Enum):
    PackageEnergyComponentOnly = 1
    DRAMEnergyComponentOnly = 2
    CoreEnergyComponentOnly = 3
    PackageDRAMEnergyComponents = 4

class ModelOutputType(enum.Enum):
    AbsPower = 1
    DynPower = 2
    XGBoostStandalonePower = 3

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
    FeatureGroup.BPFIRQ: deep_sort(BPF_FEATURES + IRQ_FEATURES),
    FeatureGroup.CounterIRQCombined: deep_sort(COUNTER_FEAUTRES + IRQ_FEATURES),
    FeatureGroup.Basic: deep_sort(BASIC_FEATURES),
    FeatureGroup.AcceleratorOnly: deep_sort(ACCELERATE_FEATURES),
}

SingleSourceFeatures = [FeatureGroup.CounterOnly.name, FeatureGroup.CgroupOnly.name, FeatureGroup.BPFOnly.name, FeatureGroup.BPFIRQ.name, FeatureGroup.KubeletOnly.name]

def is_single_source_feature_group(fg):
    return fg.name in SingleSourceFeatures

# XGBoostRegressionTrainType
class XGBoostRegressionTrainType(enum.Enum):
    TrainTestSplitFit = 1
    KFoldCrossValidation = 2

# XGBoost Model Feature and Label Incompatability Exception
class XGBoostModelFeatureOrLabelIncompatabilityException(Exception):
    """Exception raised when a saved model's features and label is incompatable with the training data. 
    
    ...

    Attributes
    ----------
    expected_features: the expected model features
    expected_labels: the expected model labels
    actual_features: the actual model features
    actual_labels: the actual model labels
    features_incompatible: true if expected_features == actual_features else false 
    labels_incompatible: true if expected_labels == actual_labels else false
    """

    expected_features: List[str]
    expected_labels: List[str]
    actual_features: List[str]
    actual_labels: List[str]
    features_incompatible: bool
    labels_incompatible: bool


    def __init__(self, expected_features: List[str], expected_labels: List[str], received_features: List[str], received_labels: List[str], message="expected features/labels are the not the same as the features/labels of the training data") -> None:
        self.expected_features = expected_features
        self.expected_labels = expected_labels
        self.received_features = received_features
        self.received_labels = received_labels
        self.features_incompatible = self.expected_features != self.actual_features
        self.labels_incompatible = self.expected_labels != self.actual_labels
        self.message = message
        super().__init__(self.message)


# XGBoost missing Model or Model Desc Exception
class XGBoostMissingModelXOrModelDescException(Exception):
    """Exception raised when saved Model is either missing the trained XGBoost Model or the Model Description.

    ...

    Attributes
    ----------
    missing_model: model is missing
    missing_model_desc: model_desc is missing
    """

    missing_model: bool
    missing_model_desc: bool

    def __init__(self, missing_model: bool, missing_model_desc: bool, message="model is missing xor model_description is missing") -> None:
        self.missing_model = missing_model
        self.missing_model_desc = missing_model_desc
        self.message = message
        super().__init__(self.message)


EnergyComponentLabelGroups = {
    EnergyComponentLabelGroup.PackageEnergyComponentOnly: deep_sort(PACKAGE_ENERGY_COMPONENT_LABEL),
    EnergyComponentLabelGroup.DRAMEnergyComponentOnly: deep_sort(DRAM_ENERGY_COMPONENT_LABEL),
    EnergyComponentLabelGroup.CoreEnergyComponentOnly: deep_sort(CORE_ENERGY_COMPONENT_LABEL),
    EnergyComponentLabelGroup.PackageDRAMEnergyComponents: deep_sort(PACKAGE_ENERGY_COMPONENT_LABEL + DRAM_ENERGY_COMPONENT_LABEL)

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