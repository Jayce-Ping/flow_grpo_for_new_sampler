export CUDA_VISIBLE_DEVICES=0,1
# k: num_images_per_prompt
# m: num_unique_prompt_per_epoch
# Previously, some weird prime numbers were used to test functionality and turns out no bug
# However, still bug for some common cases like (2, 2, 16, 32)
# Cases passed test:
# 1. (2, 2, 16, 32),
# 2. (2, 2, 24, 48)
# 3. (2, 4, 12, 48)
# 4. (2, 4, 12, 32)

accelerate launch --num_processes 2 \
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --main_process_port 29501 \
    scripts/test_sampler_dist.py \
    --dataset dataset/ocr \
    --batch_size 2 \
    --k 24 \
    --m 48 \
    --epochs 200 \
    --seed 42 \
    --num_workers 1