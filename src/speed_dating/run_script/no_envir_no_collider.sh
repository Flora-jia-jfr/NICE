#!/usr/bin/env bash

#!/bin/bash

DATA_DIR=../../dat/SpeedDatingDat/

OUTPUT_DIR=../../res/exp2_no_envir_no_collider/

#hyper parameters are not cross validated. they are picked as what seem reasonable.
HIDDEN_DIM=250
L2=0.0001
LR=0.001
STEPS=501
N_COL=20
NET=tarnet

models=(
    tarnet
    # dragon
    tarnet_single
)

mods=(
    Mod1
    Mod2
    Mod3
    Mod4
)

dims=(
    low
    med
    high
)

for NET in ${models[@]}; do
    for i in {1..10}; do
        for mod in ${mods[@]}; do
            for dim in ${dims[@]}; do
                echo "IRM no collider train_test:"
                python3 -u main.py \
                --hidden_dim=$HIDDEN_DIM \
                --l2_regularizer_weight=$L2 \
                --lr=$LR\
                --penalty_anneal_iters=401 \
                --penalty_weight=10 \
                --steps=$STEPS \
                --mod=$mod \
                --collider=0 \
                --num_col=$N_COL\
                --dimension=$dim\
                --dat=$i \
                --net=$NET\
                --data_base_dir $DATA_DIR\
                --output_base_dir $OUTPUT_DIR\
                --all_train 0

                echo "IRM no collider all_train:"
                python3 -u main.py \
                --hidden_dim=$HIDDEN_DIM \
                --l2_regularizer_weight=$L2 \
                --lr=$LR\
                --penalty_anneal_iters=401 \
                --penalty_weight=10 \
                --steps=$STEPS \
                --mod=$mod \
                --collider=0 \
                --num_col=$N_COL\
                --dimension=$dim\
                --dat=$i \
                --net=$NET\
                --data_base_dir $DATA_DIR\
                --output_base_dir $OUTPUT_DIR\
                --all_train 1


                echo "ERM no collider train_test:"
                python3 -u main.py \
                --hidden_dim=$HIDDEN_DIM  \
                --l2_regularizer_weight=$L2 \
                --lr=$LR\
                --penalty_anneal_iters=0 \
                --penalty_weight=0.0 \
                --steps=$STEPS\
                --mod=$mod\
                --collider=0\
                --num_col=$N_COL\
                --dimension=$dim\
                --dat=$i \
                --net=$NET\
                --data_base_dir $DATA_DIR\
                --output_base_dir $OUTPUT_DIR\
                --all_train 0
                
                echo "ERM no collider all_train:"
                python3 -u main.py \
                --hidden_dim=$HIDDEN_DIM  \
                --l2_regularizer_weight=$L2 \
                --lr=$LR\
                --penalty_anneal_iters=0 \
                --penalty_weight=0.0 \
                --steps=$STEPS\
                --mod=$mod\
                --collider=0\
                --num_col=$N_COL\
                --dimension=$dim\
                --dat=$i \
                --net=$NET\
                --data_base_dir $DATA_DIR\
                --output_base_dir $OUTPUT_DIR\
                --all_train 1

            done
        done
    done
done