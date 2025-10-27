export WANDB_MODE=offline
export PROJECT="gstk-sit"

torchrun --nnodes=1 --nproc_per_node=8 train.py --model SiT-XL/2 --data-path $IMAGENET_ROOT/train --wandb