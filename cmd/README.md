# Keple Model Server entrypoint

Use kepler model server function as a standalone docker container.

```
Kepler model server entrypoint

positional arguments:
  command               The command to execute.

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Specify input file name.
  -o OUTPUT_FILENAME, --output-filename OUTPUT_FILENAME
                        Specify output file name
  -s SERVER, --server SERVER
                        Specify prometheus server.
  --interval INTERVAL   Specify query interval.
  --step STEP           Specify query step.
  --metric-prefix METRIC_PREFIX
                        Specify metrix prefix to filter.
  -p PIPELINE_NAME, --pipeline-name PIPELINE_NAME
                        Specify pipeline name.
  --isolator ISOLATOR   Specify isolator name (none, min, profile, trainer).
  --profile PROFILE     Specify profile input (required for trainer and
                        profile isolator).
  --energy-source ENERGY_SOURCE
                        Specify energy source.
  --abs-trainers ABS_TRAINERS
                        Specify trainer names (use comma(,) as delimiter).
  --dyn-trainers DYN_TRAINERS
                        Specify trainer names (use comma(,) as delimiter).
```

## Get started

1. Deploy Kepler with Prometheus in the cluster exporting prometheus to port `:9090`

2. Run workloads and collect kepler metric

    Linux:

    ```bash
    docker run --rm -v "$(pwd)":/data --network=host quay.io/sustainable_computing_io/kepler-model-server:v0.6 query
    ```

    mac OS:

    ```bash
    docker run --rm -v "$(pwd)":/data quay.io/sustainable_computing_io/kepler-model-server:v0.6 query -s http://host.docker.internal:9090
    ```

    output of query will be saved as `output.json` by default

3. Run training pipeline

    ```bash
    docker run --rm -v "$(pwd)":/data quay.io/sustainable_computing_io/kepler-model-server:v0.6 train -i output.json
    ```

    output of trained model will be under folder `default`