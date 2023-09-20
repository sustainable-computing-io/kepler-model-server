# Kepler Model Server entrypoint

Use kepler model server function as a standalone docker container.

```
usage: main.py [-h] [-i INPUT] [-o OUTPUT] [-s SERVER] [--interval INTERVAL] [--step STEP] [--metric-prefix METRIC_PREFIX] [-p PIPELINE_NAME] [--extractor EXTRACTOR] [--isolator ISOLATOR] [--profile PROFILE] [--target-hints TARGET_HINTS] [--bg-hints BG_HINTS]
               [-e ENERGY_SOURCE] [--abs-trainers ABS_TRAINERS] [--dyn-trainers DYN_TRAINERS] [--benchmark BENCHMARK] [-ot OUTPUT_TYPE] [-fg FEATURE_GROUP] [--model-name MODEL_NAME] [--target-data TARGET_DATA] [--scenario SCENARIO] [--id ID] [--version VERSION]
               [--publisher PUBLISHER] [--include-raw INCLUDE_RAW]
               command

Kepler model server entrypoint

positional arguments:
  command               The command to execute.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Specify input file/folder name.
  -o OUTPUT, --output OUTPUT
                        Specify output file/folder name
  -s SERVER, --server SERVER
                        Specify prometheus server.
  --interval INTERVAL   Specify query interval.
  --step STEP           Specify query step.
  --metric-prefix METRIC_PREFIX
                        Specify metrix prefix to filter.
  -p PIPELINE_NAME, --pipeline-name PIPELINE_NAME
                        Specify pipeline name.
  --extractor EXTRACTOR
                        Specify extractor name (default, smooth).
  --isolator ISOLATOR   Specify isolator name (none, min, profile, trainer).
  --profile PROFILE     Specify profile input (required for trainer and profile isolator).
  --target-hints TARGET_HINTS
                        Specify dynamic workload container name hints (used by TrainIsolator)
  --bg-hints BG_HINTS   Specify background workload container name hints (used by TrainIsolator)
  -e ENERGY_SOURCE, --energy-source ENERGY_SOURCE
                        Specify energy source.
  --abs-trainers ABS_TRAINERS
                        Specify trainer names (use comma(,) as delimiter).
  --dyn-trainers DYN_TRAINERS
                        Specify trainer names (use comma(,) as delimiter).
  --benchmark BENCHMARK
                        Specify benchmark file name.
  -ot OUTPUT_TYPE, --output-type OUTPUT_TYPE
                        Specify output type (AbsPower or DynPower) for energy estimation.
  -fg FEATURE_GROUP, --feature-group FEATURE_GROUP
                        Specify target feature group for energy estimation.
  --model-name MODEL_NAME
                        Specify target model name for energy estimation.
  --target-data TARGET_DATA
                        Speficy target plot data (preprocess, estimate)
  --scenario SCENARIO   Speficy scenario
  --id ID               specify machine id
  --version VERSION     Specify model server version.
  --publisher PUBLISHER
                        Specify github account of model publisher
  --include-raw INCLUDE_RAW
                        Include raw query data
```

## Get started

1. Deploy Kepler with Prometheus in the cluster exporting prometheus to port `:9090`

2. Run workloads and collect kepler metric

    Linux:

    ```bash
    docker run --rm -v "$(pwd)/data":/data --network=host quay.io/sustainable_computing_io/kepler_model_server:v0.6 query
    ```

    mac OS:

    ```bash
    docker run --rm -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.6 query -s http://host.docker.internal:9090
    ```

    output of query will be saved as `output.json` by default

3. Run training pipeline

    ```bash
    docker run --rm -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.6 train -i output_kepler_query
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
    ├── <energy_source> # provided by --energy-source (default: rapl)
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
    docker run --rm -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.6 estimate -i output_kepler_query
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
      docker run --rm -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.6 plot --target-data preprocess
      ```

   5.2. Plot best prediction result (`estimate`)

      ```bash
      docker run --rm -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.6 plot --target-data estimate -i output_kepler_query
      ```

   5.3. Plot prediction result on specific trainer model and feature group (`estimate`)

      ```bash
      docker run --rm -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.6 plot --target-data estimate -i output_kepler_query --model-name GradientBoostingRegressorTrainer_1 --feature-group BPFOnly
      ```

   5.4. Plot prediction error comparison among feature group and trainer model (`error`)

    ```bash
    docker run --rm -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler_model_server:v0.6 plot --target-data error -i output_kepler_query
    ```

    output will be under the folder `output`.

5. Export

    ```bash
    KEPLER_MODEL_DB_MODELS_PATH= < path to kepler-model-db/models >
    MACHINE_ID= < machine id >
    GH_ACCOUNT= < github account >
    docker run --rm -v "${KEPLER_MODEL_DB_MODELS_PATH}":/output -v "$(pwd)/data":/data quay.io/sustainable_computing_io/kepler-model-server:v0.6 export $MACHINE_ID /output $GH_ACCOUNT
    ```

