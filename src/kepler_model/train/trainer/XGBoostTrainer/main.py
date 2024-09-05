import datetime
import json
import os
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold, cross_val_score, train_test_split

from kepler_model.train.extractor.extractor import DefaultExtractor
from kepler_model.util.train_types import (
    EnergyComponentLabelGroup,
    EnergyComponentLabelGroups,
    FeatureGroup,
    FeatureGroups,
    XGBoostMissingModelXOrModelDescException,
    XGBoostModelFeatureOrLabelIncompatabilityException,
    XGBoostRegressionTrainType,
)


# Currently Cgroup Metrics are not exported
class XGBoostRegressionStandalonePipeline:
    def __init__(self, train_type: XGBoostRegressionTrainType, save_location: str, node_level: bool) -> None:
        self.model = None
        self.train_type = train_type
        self.model_class = "xgboost"
        self.energy_source = "rapl-sysfs"
        self.feature_group = FeatureGroup.BPFIRQ
        self.save_location = save_location
        self.energy_components_labels = EnergyComponentLabelGroups[EnergyComponentLabelGroup.PackageEnergyComponentOnly]
        self.features = FeatureGroups[self.feature_group]
        print(self.energy_components_labels)
        self.model_labels = ["total_package_power"]
        # Define key model names
        if node_level:
            self.model_name = XGBoostRegressionStandalonePipeline.__name__ + "_" + "Node_Level" + "_" + self.train_type.name
        else:
            self.model_name = XGBoostRegressionStandalonePipeline.__name__ + "_" + "Container_Level" + "_" + self.train_type.name
        self.node_level = node_level
        self.initialize_relevant_models()

    # This pipeline is responsible for initializing one or more needed models
    # (Can vary depending on variables like isolator)
    def initialize_relevant_models(self) -> None:
        # Generate models and store in dict
        self.model = XGBoostRegressionModelGenerationPipeline(self.features, self.model_labels, self.save_location, self.model_name)

    def _generate_clean_model_training_data(self, extracted_data: pd.DataFrame) -> pd.DataFrame:
        # Merge Package data columns
        # Retrieve columns that end in "package_power"
        cloned_extracted_data = extracted_data.copy()
        # Column names should always be a string
        package_power_dataset = pd.DataFrame()
        for _, col_name in enumerate(cloned_extracted_data):
            if col_name.endswith("_package_power"):
                package_power_dataset[col_name] = cloned_extracted_data[col_name]
        # Sum to form total_package_power
        package_power_dataset[self.model_labels[0]] = package_power_dataset.sum(axis=1)
        # Append total_package_power to cloned_extracted_data
        cloned_extracted_data[self.model_labels[0]] = package_power_dataset[self.model_labels[0]]
        # Return cloned_extracted_data
        return cloned_extracted_data

    def train(self, prom_client=None, refined_results=None) -> None:
        results = dict()
        if prom_client != None:
            prom_client.query()
            results = prom_client.snapshot_query_result()
        elif refined_results != None:
            results = refined_results
        else:
            raise Exception("no data provided")
        # results can be used directly by extractor.py
        extractor = DefaultExtractor()
        # Train all models with extractor
        extracted_data, _, _, _ = extractor.extract(
            results, self.energy_components_labels, self.feature_group.name, self.energy_source, node_level=self.node_level
        )

        if extracted_data is not None:
            clean_df = self._generate_clean_model_training_data(extracted_data)
            self.model.train(self.train_type, clean_df)
        else:
            raise Exception("extractor failed")

    # Accepts JSON Input with feature and corresponding prediction
    def predict(self, features_and_predictions: list[dict[str, float]]) -> tuple[list[float], dict[Any, Any]]:
        # features Convert to List[List[float]]
        list_of_predictions = []
        for prediction in features_and_predictions:
            feature_values = []
            for feature in self.features:
                feature_values.append(prediction[feature])
            list_of_predictions.append(feature_values)
        return self.model.predict(list_of_predictions)


# XGBoost (Gradient Boosting Regressor) Base Model Generation
class XGBoostRegressionModelGenerationPipeline:
    """A class used to handle XGBoost Regression Model Incremental Training. This class currently only handles numerical features.

    ...

    Attributes
    ----------
    TODO

    Methods
    -------
    TODO

    """

    feature_names: list[str]
    label_names: list[str]
    model_name: str

    def __init__(self, feature_names_in_order: list[str], label_names_in_order: list[str], save_location: str, model_name: str) -> None:
        # model data will be generated consistently using the list of feature names and labels (Order does not matter)

        self.feature_names = feature_names_in_order.copy()
        self.feature_names.sort()
        self.label_names = label_names_in_order.copy()
        self.label_names.sort()
        # allow save_location to be modified
        self.save_location = save_location
        self.model_name = model_name
        self.model_filename = model_name + ".model"
        self.model_desc = "model_desc.json"

    @staticmethod
    def _generate_base_model() -> xgb.XGBRegressor:
        # n_estimators, max_depth, eta (learning rate)
        return xgb.XGBRegressor(n_estimators=1000, learning_rate=0.1)

    def _generate_model_data_filepath(self) -> str:
        return os.path.join(self.save_location, self.model_name + "_package")

    def _model_data_filepath_exists(self) -> str:
        filepath = self._generate_model_data_filepath()
        return os.path.exists(filepath)

    def model_exists(self) -> bool:
        filename_path = self._generate_model_data_filepath()
        return os.path.exists(os.path.join(filename_path, self.model_filename))

    def model_json_data_exists(self) -> bool:
        filename_path = self._generate_model_data_filepath()
        return os.path.exists(os.path.join(filename_path, self.model_desc))

    def retrieve_all_model_data(self) -> tuple[xgb.XGBRegressor | None, dict[Any, Any] | None]:
        # Note that when generating base model, it does not need to contain default hyperparameters if it will just be
        # used for prediction
        # Returns model and model_desc
        filename_path = self._generate_model_data_filepath()
        new_model = self._generate_base_model()
        if (not self.model_exists()) ^ (not self.model_json_data_exists()):
            raise XGBoostMissingModelXOrModelDescException(missing_model=self.model_exists(), missing_model_desc=self.model_json_data_exists())
        if self.model_exists() and self.model_json_data_exists():
            new_model.load_model(os.path.join(filename_path, self.model_filename))
            with open(os.path.join(filename_path, self.model_desc)) as f:
                json_data = json.load(f)
            if json_data["feature_names"] != self.feature_names or json_data["label_names"] != self.label_names:
                raise XGBoostModelFeatureOrLabelIncompatabilityException(
                    json_data["feature_names"], json_data["label_names"], self.feature_names, self.label_names
                )
            return new_model, json_data
        return None, None

    def _save_model(self, model: xgb.XGBRegressor, model_desc: dict[Any, Any]) -> None:
        filename_path = self._generate_model_data_filepath()
        if not self._model_data_filepath_exists():
            os.makedirs(filename_path)

        model.save_model(os.path.join(filename_path, self.model_filename))
        if "feature_names" not in model_desc:
            model_desc["feature_names"] = self.feature_names
        if "label_names" not in model_desc:
            model_desc["label_names"] = self.label_names
        print(model_desc)

        with open(os.path.join(filename_path, self.model_desc), "w") as f:
            json.dump(model_desc, f)

    def _clone_and_clean_model_data(self, model_data: pd.DataFrame) -> pd.DataFrame:
        # Create df with relevant feature names and label names
        new_df = pd.DataFrame()
        for feature in self.feature_names:
            new_df[feature] = model_data[feature]
        for label in self.label_names:
            new_df[label] = model_data[label]
        print(new_df.columns.tolist())
        new_df.dropna(inplace=True)
        return new_df

    def train(self, train_type: XGBoostRegressionTrainType, model_data: pd.DataFrame) -> None:
        # train_type must contain feature_names columns and label_names columns
        retrieved_model, retrieved_model_desc = self.retrieve_all_model_data()
        all_model_data_exists = retrieved_model is not None and retrieved_model_desc is not None
        print(all_model_data_exists)
        cleaned_model_data = self._clone_and_clean_model_data(model_data)
        # Either cross evaluate or perform simple test and train (Include more training styles if necessary in the future)
        # Note: We can use GridSearchCV to acquire the best hyperparameters (possible automatic feature in the future)
        if train_type.value == XGBoostRegressionTrainType.TrainTestSplitFit.value:
            self.__perform_train_test_split(all_model_data_exists, cleaned_model_data)
        elif train_type.value == XGBoostRegressionTrainType.KFoldCrossValidation.value:
            self.__perform_kfold_train(all_model_data_exists, cleaned_model_data)

    def __perform_train_test_split(self, all_model_data_exists: bool, ready_model_data: pd.DataFrame) -> None:
        # Generate new model
        new_model = self._generate_base_model()
        X = ready_model_data.loc[:, self.feature_names].values
        y = ready_model_data.loc[:, self.label_names].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
        if all_model_data_exists:
            # fit old model with new data
            old_model_filepath = os.path.join(self._generate_model_data_filepath(), self.model_filename)
            new_model.fit(X_train, y_train, xgb_model=old_model_filepath)
        else:
            # fit model from scratch
            new_model.fit(X_train, y_train)
        # Evaluate Results and store in model_desc.json
        y_predictions = new_model.predict(X_test)

        results = [value for value in y_predictions]

        rmse_res = mean_squared_error(y_test, results, squared=False)
        mae_res = mean_absolute_error(y_test, results)
        r2_res = r2_score(y_test, results)
        mape_res = mean_absolute_percentage_error(y_test, results)
        # Results
        print(f"rmse: {rmse_res}")
        print(f"r2: {r2_res}")
        print(f"mae: {mae_res}")
        print(f"mape:{mape_res}")
        print(y)
        print(np.mean(y))
        # TODO: Determine acceptable rmse, and r2 values?
        # Note that this feature could also be implemented in kepler-models
        # Generate new model_desc.json
        model_data = {
            "timestamp": datetime.datetime.now().timestamp(),
            "train_type": "fit_train_and_predict_with_test",
            "rmse": rmse_res,
            "r2": r2_res,
            "mae": mae_res,
            "mape": mape_res,
            "average_y_val": np.mean(y),
        }
        # Save Model and model_desc.json
        self._save_model(new_model, model_data)

    def __perform_kfold_train(self, all_model_data_exists: bool, ready_model_data: pd.DataFrame) -> None:
        # Generate new model
        new_model = self._generate_base_model()
        X = ready_model_data.loc[:, self.feature_names].values
        y = ready_model_data.loc[:, self.label_names].values

        # In the future, contemplate including RepeatedKFold
        kFoldCV = RepeatedKFold(n_splits=10, n_repeats=3, random_state=10)
        old_model_filepath = os.path.join(self._generate_model_data_filepath(), self.model_filename)

        if all_model_data_exists:
            params = {
                "xgb_model": old_model_filepath,
            }
            mae_cv_scores = cross_val_score(estimator=new_model, X=X, y=y, scoring="neg_mean_absolute_error", cv=kFoldCV, fit_params=params)
            mae_cv_scores = np.absolute(mae_cv_scores)
            mape_cv_scores = cross_val_score(estimator=new_model, X=X, y=y, scoring="neg_mean_absolute_percentage_error", cv=kFoldCV)
            mape_cv_scores = np.absolute(mape_cv_scores)
            r2_cv_scores = cross_val_score(estimator=new_model, X=X, y=y, scoring="r2", cv=kFoldCV)
            r2_cv_scores = np.absolute(r2_cv_scores)
        else:
            mae_cv_scores = cross_val_score(estimator=new_model, X=X, y=y, scoring="neg_mean_absolute_error", cv=kFoldCV)
            mae_cv_scores = np.absolute(mae_cv_scores)
            mape_cv_scores = cross_val_score(estimator=new_model, X=X, y=y, scoring="neg_mean_absolute_percentage_error", cv=kFoldCV)
            mape_cv_scores = np.absolute(mape_cv_scores)
            r2_cv_scores = cross_val_score(estimator=new_model, X=X, y=y, scoring="r2", cv=kFoldCV)
            r2_cv_scores = np.absolute(r2_cv_scores)
        print(f"average_mae: {mae_cv_scores.mean()}")
        print(f"mae_std: {mae_cv_scores.std()}")
        print(mape_cv_scores)
        print(f"average_mape: {mape_cv_scores.mean()}")
        print(f"mape_std: {mape_cv_scores.std()}")
        print(y)
        print(np.mean(y))
        model_data = {
            "timestamp": datetime.datetime.now().timestamp(),
            "train_type": "cross_val_score_Repeated3KFold",  # this should be variable instead of hard coded
            "average_mae": mae_cv_scores.mean(),
            "mae_std": mae_cv_scores.std(),
            "average_mape": mape_cv_scores.mean(),
            "mape_std": mape_cv_scores.std(),
            "average_r2": r2_cv_scores.mean(),
            "r2_std": r2_cv_scores.std(),
            "average_y_val": np.mean(y),
        }

        # TODO: Determine acceptable average_mae?
        # Note that this feature could also be implemented in kepler-models

        if all_model_data_exists:
            new_model.fit(X, y, xgb_model=old_model_filepath)
        else:
            # If Model is acceptable, fit it from scratch
            new_model.fit(X, y)
        print(new_model.objective)
        # Save Model and model_desc.json
        self._save_model(new_model, model_data)

    # Receives list of features and returns a list of predictions in order
    # Return None if no available model
    def predict(self, input_values: list[list[float]]) -> tuple[list[float] | None, dict[Any, Any] | None]:
        retrieved_model, retrieved_model_desc = self.retrieve_all_model_data()
        predicted_results = []
        if retrieved_model is not None:
            for feature_input in input_values:
                predict_features = np.asarray([feature_input])
                predicted_result = retrieved_model.predict(predict_features)
                predicted_results.append(predicted_result.astype(float)[0])
            return predicted_results, retrieved_model_desc
        return None, None
