#!/bin/bash
# CUDA_VISIBLE_DEVICES=3 python CDC_train_test.py --config_file ./options/realSR_EDSR.py --gpus 1 --train_file CDC_train_test.py 2>&1 | tee logs/edsr_x4.txt
CUDA_VISIBLE_DEVICES=2,3 python CDC_train_test.py --config_file ./options/realSR_RCAN.py --gpus 2 --train_file CDC_train_test.py 2>&1 | tee logs/rcan_3Dx4.txt
# CUDA_VISIBLE_DEVICES=2,3 python CDC_train_test.py --config_file ./options/realSR_HGSR_MSHR.py --gpus 2 --train_file CDC_train_test.py 2>&1 | tee logs/cdcsr_x4.txt
# CUDA_VISIBLE_DEVICES=2,3 python CDC_train_test.py --config_file ./options/realSR_MASR.py --gpus 2 --train_file CDC_train_test.py 2>&1 | tee logs/masr_x4.txt


# python $2 --config_file $3 --gpus $4 --train_file $2 2>&1 | tee logs/$1.txt
# python -m cProfile -o result.txt $2 --config_file $3 --gpus $4 --train_file $2 2>&1 | tee logs/$1.txt

# bash ./train_pc.sh cdc_x4 ./CDC_train_test.py ./options/realSR_HGSR_MSHR.py 1
