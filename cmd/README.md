# Kepler Model Server entrypoint

Use kepler model server function as a standalone docker container.

## Get started

1. Deploy Kepler with Prometheus in the cluster exporting prometheus to port `:9090`

2. Run workloads and collect kepler metric

    Linux:

    ```bash
    docker run --rm -v "$(pwd)/data":/data --network=host quay.io/sustainable_computing_io/kepler_model_server:v0.7.11 query
    ```

    mac OS:

    ```bash
    docker run --rm -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.7.11 query -s http://host.docker.internal:9090
    ```

    output of query will be saved as `output.json` by default

3. Run training pipeline

    ```bash
    docker run --rm -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.7.11 train -i output_kepler_query
    ```

    output of trained model will be under pipeline folder `default` or can be specified by `-p`

    ```bash
    <pipeline_name>.zip # archived pipeline for model server to load
    <pipeline_name> # provided by -p, --pipeline-name (default: default)
    ├── metadata.json # pipeline metadata such as pipeline name, extractor, isolator, trainer list
    ├── preprocessed_data 
    │   ├── <feature_group>_abs_data.csv
    │   └── <feature_group>_dyn_data.csv
    ├── profile # if --profile is providied
    │   ├── acpi.json
    │   └── rapl.json
    ├── <energy_source> # provided by --energy-source (default: rapl-sysfs)
    │   ├── <model_type> # AbsPower or DynPower
    │   │   ├── <feature_group>  # e.g., BPFOnly
    │   │   │   ├── <model_name>
    │   │   │   │   ├── metadata.json # model metadata
    │   │   │   │   └── <model_files> 
    │   │   │   │   └── ...
    │   │   │   │   └── weight.json # model weight in json format if support for kepler to load
    │   │   │   ├── <model_name>.zip # archived model for estimator to load
    ├── <energy_source>_AbsPower_model_metadata.csv # AbsPower models summary
    ├── <energy_source>_DynPower_model_metadata.csv # DynPower models summary
    └── train_arguments.json
    ```

4. Test estimation

    ```bash
    docker run --rm -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.7.11 estimate -i output_kepler_query
    ```

    output will be under the folder `output`.

    ```
    output
    ├── rapl_estimation_result.csv.csv
    └── rapl_model.zip
    ```

5. Plot and save image 
  
   5.1. Plot extracted and isolated data (`preprocess`)

      ```bash
      docker run --rm -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.7.11 plot --target-data preprocess
      ```

   5.2. Plot best prediction result (`estimate`)

      ```bash
      docker run --rm -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.7.11 plot --target-data estimate -i output_kepler_query
      ```

   5.3. Plot prediction result on specific trainer model and feature group (`estimate`)

      ```bash
      docker run --rm -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.7.11 plot --target-data estimate -i output_kepler_query --model-name GradientBoostingRegressorTrainer_0 --feature-group BPFOnly
      ```

   5.4. Plot prediction error comparison among feature group and trainer model (`error`)

    ```bash
    docker run --rm -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.7.11 plot --target-data error -i output_kepler_query
    ```

    output will be under the folder `output`.

5. Export

    ```bash
    KEPLER_MODEL_DB_MODELS_PATH= < path to kepler-model-db/models >
    MACHINE_ID= < machine id >
    GH_ACCOUNT= < github account >
    docker run --rm -v "${KEPLER_MODEL_DB_MODELS_PATH}":/output -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.7.11 export $MACHINE_ID /output $GH_ACCOUNT
    ```

