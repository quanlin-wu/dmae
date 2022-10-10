## Pre-training DMAE
To pre-train DMAE-Base with **single-node training**, run the following on 1 node with 8 GPUs:
```
python -m torch.distributed.launch --nproc_per_node=8 \
    main_pretrain.py \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --batch_size 64 \
    --accum_iter 8 \
    --model dmae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --sigma 0.25 \
    --epochs 1100 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}
```
- Here the effective batch size is 64 (`batch_size` per gpu) * 8 (`accum_iter`) * 8 (gpus per node) = 4096.
- `blr` is the base learning rate. The actual `lr` is computed by the [linear scaling rule](https://arxiv.org/abs/1706.02677): `lr` = `blr` * effective batch size / 256.

Script for DMAE-Large:
```
python -m torch.distributed.launch --nproc_per_node=8 \
    main_pretrain.py \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --batch_size 64 \
    --accum_iter 8 \
    --model dmae_vit_large_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --sigma 0.25 \
    --epochs 1600 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path ${IMAGENET_DIR}
```
#### Continue Pre-training on CIFAR-10

In order to learn the dataset bias, we continue pre-training our DMAE-B model on CIFAR-10 by running the following on 1 node with 1 GPU:
```
python -m torch.distributed.launch --nproc_per_node=1 \
    pretrain_cifar10.py \
    --output_dir  ${OUTPUT_DIR} \
    --log_dir  ${OUTPUT_DIR} \
    --resume ${PRETRAIN_CHKPT} \
    --data_path ${CIFAR-10_DIR} \
    --batch_size 64 \
    --accum_iter 8 \
    --model dmae_vit_base_patch16 \
    --norm_pix_loss \
    --mask_ratio 0.75 \
    --sigma 0.25 \
    --epoch_start 0 --epochs 50 \
    --warmup_epochs 10 \
    --blr 5e-5 --weight_decay 0.05
```
- Here we use a smaller effective batch size 64 (`batch_size` per gpu) * 8 (`accum_iter`) * 1 (gpus per node) = 512.