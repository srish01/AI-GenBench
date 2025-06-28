#!/bin/bash

# Default paths
BENCHMARK_PIPELINE_CFG="training_configurations/benchmark_pipelines/base_benchmark_sliding_windows.yaml"
MODEL_CFG="training_configurations/dinov2/dinov2_tune_resize.yaml"
ADDITIONAL_CFG="local_config.yaml"
EXPERIMENT_ID="MY_EXPERIMENT_ID"

# Print the command that will be executed
echo "Executing: python lightning_main.py fit --config $BENCHMARK_PIPELINE_CFG --config $MODEL_CFG --config $ADDITIONAL_CFG --experiment_info.experiment_id $EXPERIMENT_ID $@"

# Run the training script with the provided or default paths
python lightning_main.py fit --config $BENCHMARK_PIPELINE_CFG --config $MODEL_CFG --config $ADDITIONAL_CFG --experiment_info.experiment_id $EXPERIMENT_ID "$@"