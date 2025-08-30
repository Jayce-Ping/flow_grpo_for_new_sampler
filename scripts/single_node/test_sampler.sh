export CUDA_VISIBLE_DEVICES=0,3,4,5
# k: num_images_per_prompt
# m: num_unique_prompt_per_epoch
# Set some weird numbers here to test its functionality
accelerate launch --num_processes 4 \
    --config_file scripts/accelerate_configs/deepspeed_zero2.yaml \
    --main_process_port 29501 \
    scripts/test_sampler_dist.py \
    --dataset dataset/ocr \
    --batch_size 3 \
    --k 23 \
    --m 11 \
    --epochs 200 \
    --seed 42 \
    --num_workers 1