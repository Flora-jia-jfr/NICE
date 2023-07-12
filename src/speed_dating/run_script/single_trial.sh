#!/usr/bin/env bash

#!/bin/bash

DATA_DIR=../../dat/SpeedDatingDat/

OUTPUT_DIR=../../res/exp2-single_trial/

#hyper parameters are not cross validated. they are picked as what seem reasonable.
HIDDEN_DIM=250
L2=0.0001
LR=0.001
STEPS=501
N_COL=20
# NET=tarnet
# NET=dragon
NET=tarnet_single

echo "IRM no Collider:"
python3 -u main.py \
    --hidden_dim=$HIDDEN_DIM \
    --l2_regularizer_weight=$L2 \
    --lr=$LR \
    --penalty_anneal_iters=401 \
    --penalty_weight=10 \
    --steps=$STEPS \
    --mod="Mod1"\
    --collider=0\
    --num_col=$N_COL\
    --dimension="high"\
    --dat=1 \
    --net=$NET\
    --data_base_dir $DATA_DIR\
    --output_base_dir $OUTPUT_DIR
