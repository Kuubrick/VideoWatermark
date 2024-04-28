CUDA_VISIBLE_DEVICES=0 python inference.py \
    --num_frame 8 \
    --fps 8 \
    --prompt_path datas/fetv_data.json \
    --output_root_path output_videos