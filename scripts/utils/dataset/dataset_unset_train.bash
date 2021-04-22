#!/usr/bin/env bash

owner="googleresearch"
train_model_dir="models_train"

if [[ -d "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models" ]]; then
    if [[ ! -d "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/$train_model_dir" ]]; then
        echo "Info: Unsetting all models under "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models" and moving them to temporary directory "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/$train_model_dir". Use 'ros2 run drl_grasping dataset_set_train.bash' to reactivate"
        mv "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models" "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/$train_model_dir"
    else
        echo "Error: Directory "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/$train_model_dir" already exists. Unsetting of training dataset skipped."
        exit 1
    fi
else
    echo "Error: Directory "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models" does not exist. Unsetting of training dataset skipped. Please make sure to download training dataset first."
    exit 1
fi
