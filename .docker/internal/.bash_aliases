#!/usr/bin/env bash
# ~/.bash_aliases: Custom aliases.

alias _nano_envs='nano ${WS_SRC_DIR}/drl_grasping/drl_grasping/envs/__init__.py'
alias _nano_ex_train='nano ${WS_SRC_DIR}/drl_grasping/examples/ex_train.bash'
alias _nano_ex_train_dreamerv2='nano ${WS_SRC_DIR}/drl_grasping/examples/ex_train_dreamerv2.bash'
alias _nano_sac='nano ${WS_SRC_DIR}/drl_grasping/hyperparams/sac.yml'
alias _nano_td3='nano ${WS_SRC_DIR}/drl_grasping/hyperparams/td3.yml'
alias _nano_tqc='nano ${WS_SRC_DIR}/drl_grasping/hyperparams/tqc.yml'
alias _train='ros2 run drl_grasping ex_train.bash'
alias _train_dreamerv2='ros2 run drl_grasping ex_train_dreamerv2.bash'
