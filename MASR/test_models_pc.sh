#!/bin/bash
CUDA_VISIBLE_DEVICES=3 python CDC_test.py --result_dir 3Dx4 --pretrain ./models/RCAN_X4_best.pth --gpus 1 2>&1 | tee logs/TestLogs/3Dx4test.txt
# CUDA_VISIBLE_DEVICES=3 python CDC_test.py --result_dir edsr_x4test --pretrain ./models/EDSR_X4_best.pth --gpus 1 2>&1 | tee logs/TestLogs/edsr_x4test.txt
# python $2 --result_dir $1 --pretrain $3 --gpus $4 2>&1 | tee logs/TestLogs/$1.txt

# bash test_models_pc.sh cdc_x4_test ./CDC_test.py ./models/HGSR-MHR_X4_SubRegion_GW_283.pth 1
