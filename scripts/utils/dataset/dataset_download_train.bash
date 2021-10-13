#!/usr/bin/env bash

owner="googleresearch"
model_names="3d_dollhouse_happy_brother
             ecoforms_plant_container_urn_55_mocha
             3d_dollhouse_lamp
             f5_trx_fg
             3d_dollhouse_refrigerator
             fire_engine
             3d_dollhouse_sink
             folic_acid
             3d_dollhouse_swing
             grandfather_doll
             3d_dollhouse_tablepurple
             granimals_20_wooden_abc_blocks_wagon
             3m_vinyl_tape_green_1_x_36_yd
             heavyduty_flashlight
             45oz_ramekin_asst_deep_colors
             imaginext_castle_ogre
             ace_coffee_mug_kristen_16_oz_cup
             lacing_sheep
             air_hogs_wind_flyers_set_airplane_red
             markings_letter_holder
             android_figure_panda
             mens_santa_cruz_thong_in_tan_r59c69darph
             animal_planet_foam_2headed_dragon
             mini_excavator
             baby_elements_stacking_cups
             mini_roller
             balancing_cactus
             my_little_pony_princess_celestia
             bia_porcelain_ramekin_with_glazed_rim_35_45_oz_cup
             nickelodeon_teenage_mutant_ninja_turtles_leonardo
             big_dot_aqua_pencil_case
             nikon_1_aw1_w11275mm_lens_silver
             black_and_decker_tr3500sd_2slice_toaster
             nintendo_yoshi_action_figure
             blackblack_nintendo_3dsxl
             nordic_ware_original_bundt_pan
             bradshaw_international_11642_7_qt_mp_plastic_bowl
             olive_kids_birdie_munch_n_lunch
             breyer_horse_of_the_year_2015
             peekaboo_roller
             brisk_iced_tea_lemon_12_12_fl_oz_355_ml_cans_144_fl_oz_426_lt
             playmates_industrial_cosplinter_teenage_mutant_ninja_turtle_action_figure
             brother_lc_1053pks_ink_cartridge_cyanmagentayellow_1pack
             playmates_nickelodeon_teenage_mutant_ninja_turtles_shredder
             bunny_racer
             retail_leadership_summit_ect3zqhyikx
             calphalon_kitchen_essentials_12_cast_iron_fry_pan_black
             room_essentials_mug_white_yellow
             canon_225226_ink_cartridges_blackcolor_cyan_magenta_yellow_6_count
             schleich_allosaurus
             chefmate_8_frypan
             schleich_hereford_bull
             chelsea_blkheelpmp_dwxltznxlzz
             schleich_lion_action_figure
             chicken_nesting
             schleich_spinosaurus_action_figure
             circo_fish_toothbrush_holder_14995988
             schleich_therizinosaurus_ln9cruulpqc
             closetmaid_premium_fabric_cube_red
             shaxon_100_molded_category_6_rj45rj45_shielded_patch_cord_white
             coast_guard_boat
             spiderman_titan_hero_12inch_action_figure_5hnn4mtkfsp
             cole_hardware_antislip_surfacing_material_white
             stacking_bear
             colton_wntr_chukka_y4jo0i8jqfw
             stacking_ring
             craftsman_grip_screwdriver_phillips_cushion
             thomas_friends_wooden_railway_talking_thomas_z7yi7ufhjrj
             crazy_shadow_2
             threshold_porcelain_pitcher_white
             crosley_alarm_clock_vintage_metal
             victor_reversible_bookend
             digital_camo_double_decker_lunch_bag
             weston_no_33_signature_sausage_tonic_12_fl_oz
             dino_3
             whale_whistle_6pcs_set
             dino_5
             wooden_abc_123_blocks_50_pack
             ecoforms_plant_container_qp6coral
             xyli_pure_xylitol"

if [[ -d "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models" ]]; then
    echo "Error: There are already some models inside "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models". Please move them to a temporary location before continuing. If you have downloaded testing dataset bofere, please use 'ros2 run drl_grasping dataset_unset_test.bash' ('ros2 run drl_grasping set_dataset_test.bash' to reactivate it)."
    exit 1
fi

for model_name in $model_names; do
    if [[ ! -d "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models/$model_name" ]]; then
        model_uri="https://fuel.ignitionrobotics.org/1.0/$owner/models/$model_name"
        echo "Info: Downloading model '$model_uri'"
        ign fuel download -t model -u "$model_uri"
    fi
done
for job in $(jobs -p); do
    wait $job
    echo "Info: Model downloaded"
done

echo "Info: All models downloaded. Note: Downloaded models are combined with whatever models were already inside "$HOME/.ignition/fuel/fuel.ignitionrobotics.org/$owner/models" and all of these will be treated as training dataset."
