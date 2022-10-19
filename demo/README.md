## Demo Data
This folder includes offline collected data, trained initial model and predicted results.
- `query_data` 
  - **coremark_4threads_to_32threads_5rep_ondemand_demo:** data from coremark benchmark varied thread from 1, 4, 8, ..., 32 for 5 repetition
  - **idle_1000s:** data collected in idle state for 1000s
- `initial_model_train.py`
  - run to train data from query_data folder (default: coremark_4threads_to_32threads_5rep_ondemand_demo) using KerasCompFullPipeline pipeline
  - the trained model is saved in `demo/models/models/AbsComponentPower/Full/KerasCompFullPipeline`
- `test_predict.py `
  - apply trained model with collected data and save predicted results in `predicted_data/<data>.csv`
- `predicted_data/core.txt` and `predicted_data/dram.txt` are MAE over epoch for each trained component.
