#!/usr/bin/env bash

# Change ngpus to match the number of GPUs available.
# If run out of GPU memory, increase "--batch-split" argument.

# get the data
#bash get_data.sh
#mkdir -p checkpoints

ngpus=$2
args="
--data data/enwik8 \
--nlayers 12 \
--hid-sz 512 \
--inner-hid-sz 2048 \
--nheads 8 \
--attn-span 8192 \
--block-sz 512 \
--batch-sz 64 \
--lr 0.07 \
--momentum 0 \
--dropout 0.3 \
--optim adagrad \
--lr-warmup 32000 \
--grad-clip 0.03 \
--niter 600 \
--nbatches 1000 \
--adapt-span \
--adapt-span-loss 0.0000005 \
--adapt-span-cache \
--shared-ffn \
--shared-attn \
--distributed \
--checkpoint checkpoints/enwik8_sharedall_512.pt
"

export CUDA_VISIBLE_DEVICES=$1

echo "Training ..."
# using the pytorch distributed launching
python3 -m torch.distributed.launch --master_port 1002 --nproc_per_node=$ngpus main.py $args


#echo "Evaluation ..."
## use a smaller batch size to reduce tokens without context and omitted tokens.
#python3 -m torch.distributed.launch --nproc_per_node=$ngpus main.py $args \
#  --full-eval-mode --batch-sz 8
