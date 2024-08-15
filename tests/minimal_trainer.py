from pipeline_test import process
from kepler_model.util import FeatureGroup

trainer_names = ["GradientBoostingRegressorTrainer", "SGDRegressorTrainer", "XgboostFitTrainer"]
valid_feature_groups = [FeatureGroup.BPFOnly]

if __name__ == "__main__":
    process(target_energy_sources=["acpi", "rapl-sysfs"], abs_trainer_names=trainer_names, dyn_trainer_names=trainer_names, valid_feature_groups=valid_feature_groups)
