### Fine-tuning on ImageNet
To fine-tune our pre-trained ViT-Base with **single-node training**, run the following on 1 node with 8 GPUs:
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --sigma ${SIGMA} \
    --con_reg --reg_lbd ${LBD} --reg_eta ${ETA} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR}
```
- Here the effective batch size is 32 (`batch_size` per gpu) * 4 (`accum_iter`) * 8 (gpus) = 1024. `--accum_iter 4` simulates 4 nodes.
- In default we use `reg_lbd` = 2.0, `reg_eta` = 0.5 for `sigma` in {0.25, 0.5}, and use `reg_lbd` = 2.0, `reg_eta` = 0.1 for `sigma` = 1.0.

Script for fine-tuning pretrained ViT-Large:
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_large_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --sigma ${SIGMA} \
    --epochs 50 \
    --blr 1e-3 --layer_decay 0.75 \
    --weight_decay 0.05 --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```

To fine-tuning by **RS method**(only use CrossEntropy classification loss):
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_finetune.py \
    --accum_iter 4 \
    --batch_size 32 \
    --model vit_base_patch16 \
    --finetune ${PRETRAIN_CHKPT} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --sigma ${SIGMA} \
    --epochs 100 \
    --blr 5e-4 --layer_decay 0.65 \
    --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 \
    --dist_eval --data_path ${IMAGENET_DIR}
```

### Linear-probing
Run the following on 1 nodes with 8 GPUs:
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 main_linprobe.py \
    --accum_iter 4 \
    --batch_size 512 \
    --model vit_base_patch16 --cls_token \
    --finetune ${PRETRAIN_CHKPT} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --sigma ${SIGMA} \
    --epochs 90 \
    --blr 0.1 \
    --weight_decay 0.0 \
    --dist_eval --data_path ${IMAGENET_DIR}
```
- Here the effective batch size is 512 (`batch_size` per gpu) * 4 (`accum_iter`) * 8 (gpus per node) = 16384.


### Fine-tuning on CIFAR-10
To fine-tune our ViT-Base model on CIFAR-10, run the following on 1 node with 1 GPU:
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=1 finetune_cifar10.py \
    --accum_iter 4 \
    --batch_size 64 \
    --model vit_base_patch16 \
    --data_path ${CIFAR-10_DIR} \
    --finetune ${PRETRAIN_CHKPT} \
    --output_dir ${OUTPUT_DIR} \
    --log_dir ${OUTPUT_DIR} \
    --sigma ${SIGMA} \
    --con_reg --reg_lbd ${LBD} --reg_eta ${ETA} \
    --epochs 50 \
    --blr 5e-4 --layer_decay 0.65 \
    --dist_eval --weight_decay 0.05 --drop_path 0.1 \
```
- Here the effective batch size is 64 (`batch_size` per gpu) * 4 (`accum_iter`) * 1 (gpus) = 256.
- In default we use `reg_lbd` = 2.0, `reg_eta` = 0.5 for `sigma` in {0.25, 0.5}, and use `reg_lbd` = 2.0, `reg_eta` = 0.1 for `sigma` = 1.0. (The same with fine-tuning on ImageNet)


### Evaluation
To evaluate the certified accuracies of ViT-B/L on **ImageNet**:
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    certify.py --eval \
    --resume ${MODEL_PTH} \
    --model vit_base_patch16 \
    --sigma ${SIGMA} \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --sample_interval 50
```
Script for evaluating linear-probing:
```
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=8 \
    certify.py --eval \
    --resume ${MODEL_PTH} \
    --model vit_base_patch16 --cls_token --linprobe\
    --sigma ${SIGMA} \
    --dist_eval --data_path ${IMAGENET_DIR} \
    --sample_interval 50
```
- The result is averaged over 1,000(**testset_size** / `sample_interval`) images uniformly selected from ImageNet validation set.
- If `sigma` is not specified, the certified accuracies with Gaussian magnitude **{0.25, 0.5, 1.0}**  will be both evaluated.

Evaluation on **CIFAR-10**:
```
OMP_NUM_THREADS=1 python3 -m torch.distributed.launch --nproc_per_node=1 \
    certify_cifar10.py --eval \
    --resume ${MODEL_PTH} \
    --model vit_base_patch16 \
    --dist_eval  \
    --sample_interval 1 \
    --nb_classes 10 \
    --sigma ${SIGMA} \
    --num 100000
```
- Here we draw `num = 100, 000` noise samples and report results averaged over the entire CIFAR-10 test set. 