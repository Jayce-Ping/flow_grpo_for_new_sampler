# export CUDA_VISIBLE_DEVICES=0,3,4,5
# k: num_images_per_prompt
# m: num_unique_prompt_per_epoch

accelerate launch --num_processes 8 \
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --main_process_port 29501 \
    scripts/test_sampler_dist.py \
    --dataset dataset/ocr \
    --batch_size 8 \
    --k 12 \
    --m 32 \
    --epochs 200 \
    --seed 42 \
    --num_workers 1

# Cases passed test:
# 2 gpus:
#   1. (2, 2, 16, 32),
#   2. (2, 2, 24, 48)
#   3. (2, 4, 12, 48)
#   4. (2, 4, 12, 32)
# 4 gpus:
#   1. (4, 2, 24, 48)
#   2. (4, 3, 16, 48)
#   3. (4, 5, 17, 23)
#   4. (4, 9, 18, 24)
# 8 gpus:
#   1. (8, 9, 18, 48)
#   2. (8, 8, 12, 48)
#   3. (8, 9, 12, 96)
#   4. (8, 8, 12, 32)
