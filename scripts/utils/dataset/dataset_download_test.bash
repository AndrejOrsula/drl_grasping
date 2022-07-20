#!/usr/bin/env bash

COLLECTION_OWNER="googleresearch"
OWNER_DIR="${HOME}/.ignition/fuel/fuel.ignitionrobotics.org/${COLLECTION_OWNER}"
MODELS_TEST_DIR="models_test"

MODEL_NAMES=(
    3d_dollhouse_sofa
    dino_4
    nintendo_mario_action_figure
    android_figure_orange
    dpc_handmade_hat_brown
    olive_kids_robots_pencil_case
    bia_cordon_bleu_white_porcelain_utensil_holder_900028
    germanium_ge132
    schleich_african_black_rhino
    central_garden_flower_pot_goo_425
    grandmother
    schleich_s_bayala_unicorn_70432
    chelsea_lo_fl_rdheel_zaqrnhlefw8
    kong_puppy_teething_rubber_small_pink
    school_bus
    cole_hardware_butter_dish_square_red
    lenovo_yoga_2_11
    weisshai_great_white_shark
    cole_hardware_school_bell_solid_brass_38
    mini_fire_engine
)

if [[ -d "${OWNER_DIR}/models" ]]; then
    echo >&2 "Error: There are already some models inside '${OWNER_DIR}/models'. Please move them to a temporary location before continuing."
    exit 1
fi

N_PARALLEL_DOWNLOADS=$(($(nproc) < 16 ? $(nproc) : 16))
for model_name in "${MODEL_NAMES[@]}"; do
    if [[ ! -d "${OWNER_DIR}/models/${model_name}" ]]; then
        MODEL_URI="https://fuel.ignitionrobotics.org/1.0/${COLLECTION_OWNER}/models/${model_name}"
        echo "Info: Downloading model '${MODEL_URI}'."
        ign fuel download -t model -u "${MODEL_URI}" &
    fi
    while (($(jobs -p | wc -l | tr -d 0) >= "${N_PARALLEL_DOWNLOADS}")); do
        wait -n
    done
done
for job in $(jobs -p); do
    wait "${job}"
done

mv "${OWNER_DIR}/models" "${OWNER_DIR}/${MODELS_TEST_DIR}" &&
echo "Info: All downloaded models were moved to '${OWNER_DIR}/${MODELS_TEST_DIR}'."
