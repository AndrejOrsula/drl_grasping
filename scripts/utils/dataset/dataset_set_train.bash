#!/usr/bin/env bash

owner="googleresearch"
train_model_dir="models_train"

if [[ -d "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/$train_model_dir" ]]; then
    if [[ ! -d "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models" ]]; then
        echo "Info: Setting all models under "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/$train_model_dir" and moving them to "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models"."
        mv "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/$train_model_dir" "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models"
    else
        echo "Error: Directory "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models" already exists. Setting of training dataset skipped."
        exit 1
    fi
else
    echo "Error: Directory "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/$train_model_dir" does not exist. Setting of training dataset skipped. Please make sure to download training dataset first."
    echo "Info: Setting of dataset has only an effect if the dataset was previously unset."
    exit 1
fi
