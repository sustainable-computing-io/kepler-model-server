# Kepler Model Server
**Online ML model Server for Kepler (Developer Readme)**

**Different ways to run the model training on the server:**

**Testing the models using simple pre-written test cases for model training error checking:**

From the project folder, run 'python -m tests.kepler_regression_model_tests' to create, train, and save Dram and Core Models or retrain and save pre-existing Dram and Core Models.

**Running the model server with Kepler metrics (Online Learning)**

Set up and run [Kepler](https://github.com/sustainable-computing-io/kepler). [Kepler](https://github.com/sustainable-computing-io/kepler) will now export Prometheus metrics. Run the model server's energy_scheduler.py module to pull [Kepler's](https://github.com/sustainable-computing-io/kepler) Prometheus metrics. These metrics will be used to automatically train a new model or retrain a model. Currently only the core_model Linear Regression Model is included.  

**Viewing the exported model weights for a given model:**

**Checking the model weights for a given model using Flask Routes:**

To test the routes that use the model, call 'python server.py'. Available routes include '/models/<model_type>' and '/model-weights/<model_type>'. '/models/<model_type>' returns the saved model as a zip that needs to be extracted before being used. '/model-weights/<model_type>' returns the weights of a given model. Acceptable values for <model_type> are 'core_model' and 'dram_model'.

**Checking the model weights using Prometheus:**

    1. Call python server.py. Navigate to '/metrics' route to view all model weights for all available models. This is for quick testing purposes only

    OR

    2. Call 'docker-compose up' at project root. Navigate to '/metrics' route to view all model weights for all available models. 

**Checking the model weights using OpenTelemetry (WIP):**

Call 'docker-compose up' at project root. Prometheus metrics will be scraped and converted to OpenTelemetry Metrics. These metrics will be shown in the log and will be exported to a otlp endpoint. A Honeycomb.io dataset is currently in the works.

