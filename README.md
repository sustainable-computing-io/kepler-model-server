# Kepler Model Server
Online ML model Server for Kepler

Launching using Python with test files

From the project folder, run 'python -m tests.kepler_regression_model_tests' to create, train, and save Dram and Core Models or retrain and save pre-existing Dram and Core Models.

To test the routes that use the model, call python server.py. Available routes include '/models/<model_type>' and '/model-weights/<model_type>'. '/models/<model_type>' returns the saved model as a zip that needs to be extracted before being used. '/model-weights/<model_type>' returs the weights of a given model. Acceptable values for <model_type> are 'core_model' and 'dram_model'.

Test tfio.experimental.IODataset.from_prometheus()

Run './prometheus' with the given prometheus.yml file. Then run 'energy_scheduler.py' by calling 'python energy_scheduler.py'. The script will wait 5 seconds before scraping from prometheus endpoint just to make 5 seconds worth of data is retrieved. 