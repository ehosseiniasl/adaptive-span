#!/usr/bin/env bash

# get the data
#bash get_data.sh
#mkdir -p checkpoints

# Change ngpus to match the number of GPUs available.
# If run out of GPU memory, increase "--batch-split" argument.

ngpus=1
args="
--data data/text8 \
--nlayers 12 \
--hid-sz 512 \
--inner-hid-sz 2048 \
--nheads 8 \
--attn-span 8192 \
--block-sz 256 \
--batch-sz 64 \
--lr 0.07 \
--momentum 0 \
--dropout 0.3 \
--optim adagrad \
--lr-warmup 32000 \
--grad-clip 0.03 \
--niter 1000 \
--nbatches 500 \
--adapt-span \
--adapt-span-loss 0.0000005 \
--adapt-span-cache \
--batch-split 8 \
--checkpoint checkpoints/text8_256.pt
"

#--distributed \

echo "Training ..."
# using the pytorch distributed launching
#python3 -m torch.distributed.launch --nproc_per_node=$ngpus main.py $args
CUDA_VISIBLE_DEVICES=$1 python3 main.py $args
#&> text8_256.log

#tail -f text8_256.log

#echo "Evaluation ..."
# use a smaller batch size to reduce tokens without context and omitted tokens.
#python3 -m torch.distributed.launch --nproc_per_node=$ngpus main.py $args \
#  --full-eval-mode --batch-sz 8


