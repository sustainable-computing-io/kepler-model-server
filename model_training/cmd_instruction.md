# Manual Metric Collection and Training with Entrypoint

## 1. Collect metrics
Without benchmark/pipeline automation, kepler metrics can be collected by `query` function by either one of the following options.
### 1.1. by defining start time and end time

```bash
# value setting
BENCHMARK= # name of the benchmark (will generate [BENCHMARK].json to save start and end time for reference)
PROM_URL= # e.g., http://localhost:9090
START_TIME= # format date +%Y-%m-%dT%H:%M:%SZ
END_TIME= # format date +%Y-%m-%dT%H:%M:%SZ
COLLECT_ID= # any unique id e.g., machine name

# query execution
DATAPATH=/path/to/workspace python cmd/main.py query --benchmark $BENCHMARK --server $PROM_URL --output kepler_query --start-time $START_TIME --end-time $END_TIME --id $COLLECT_ID
```

### 1.2. by defining last interval from the execution time

```bash
# value setting
BENCHMARK= # name of the benchmark (will generate [BENCHMARK].json to save start and end time for reference)
PROM_URL= # e.g., http://localhost:9090
INTERVAL= # in second
COLLECT_ID= # any unique id e.g., machine name

# query execution
DATAPATH=/path/to/workspace python cmd/main.py query --benchmark $BENCHMARK --server $PROM_URL --output kepler_query --interval $INTERVAL --id $COLLECT_ID
```

### Output:
There will three files created in the `/path/to/workspace`, those are:
- `kepler_query.json`: raw prometheus query response
- `<COLLECT_ID>.json`: machine system features (spec)
- `<BENCHMARK>.json`: an item contains startTimeUTC and endTimeUTC

## 2. Train models

```bash
# value setting
PIPELINE_NAME= # any unique name for the pipeline (one pipeline can be accumulated by multiple COLLECT_ID)

# train execution
# require COLLECT_ID from collect step
DATAPATH=/path/to/workspace MODEL_PATH=/path/to/workspace python cmd/main.py train --pipeline-name $PIPELINE_NAME --input kepler_query --id $COLLECT_ID
```

## 3. Export models
Export function is to archive the model that has an error less than threshold from the trained pipeline and make a report in the format that is ready to push to kepler-model-db.

### 3.1. exporting the trained pipeline with BENCHMARK

The benchmark file is created by CPE operator or by step 1.1. or 1.2..

```bash
# value setting
EXPORT_PATH= # /path/to/kepler-model-db/models
PUBLISHER= # github account of publisher

# export execution
# require BENCHMARK from collect step
# require PIPELINE_NAME from train step
DATAPATH=/path/to/workspace MODEL_PATH=/path/to/workspace python cmd/main.py export --benchmark $BENCHMARK --pipeline-name $PIPELINE_NAME -o $EXPORT_PATH --publisher $PUBLISHER --zip=true 
```

### 3.2. exporting the trained models without BENCHMARK

If the data is collected by tekton, there is no benchmark file created. Need to manually set `--collect-date` instead of `--benchmark` parameter.

```bash
# value setting
EXPORT_PATH= # /path/to/kepler-model-db/models
PUBLISHER= # github account of publisher
COLLECT_DATE= # collect date

# export execution
# require BENCHMARK from collect step
# require PIPELINE_NAME from train step
DATAPATH=/path/to/workspace MODEL_PATH=/path/to/workspace python cmd/main.py export --pipeline-name $PIPELINE_NAME -o $EXPORT_PATH --publisher $PUBLISHER --zip=true --collect-date $COLLECT_DATE
```

