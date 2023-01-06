# Contributing

Roadmap:
- [ ] Build default pipelines for all kinds and forms of models and input types
- [ ] Increase accessibility to the model
  - [ ] Enable post-based approach to provide model weights via prometheus
        Checking the model weights using OpenTelemetry
        Call 'docker-compose up' at project root. Prometheus metrics will be scraped and converted to OpenTelemetry Metrics. These metrics will be shown in the log and will be exported to a otlp endpoint. A Honeycomb.io dataset is currently in the works.
  - [ ] Plugin Cloud object storage

The main source codes are in [server directory](./server/).
```bash
server
├── model_server.py # program endpoint for serving API routes
├── online_trainer.py # program endpoint for periodic online training
├── prom # prometheus-related functions
├── train # model definition and online training pipelines
└── util # general utility functions 
``` 

#### Implementing Pipelines
Check details in [here](./doc/train_pipeline.md)

#### Implementing test case
Check the [tests directory](./tests/)
```bash
├── tests
│   ├── test_models # initial model for testing
│   ├── download # output path when calling API endpoints
│   └── query_data # sample query data for testing the pipelines
``` 

#### Providing offline trained model
- Check model format in [here](./doc/model_format.md)
- To use pipeline framework, check offline train example in [train_test.py](./tests/train_test.py)
