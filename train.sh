## Train the LDM from scracth with a flan-t5-large text encoder

accelerate launch train.py \
--train_file="data/config_samples/train.json" --validation_file="data/config_samples/valid.json" --test_file="data/config_samples/test.json" \
--scheduler_name="ckpt/stable-diffusion-2-1" \
--unet_model_config="configs/diffusion_model_config_large_2048.json" --freeze_text_encoder \
--gradient_accumulation_steps 4 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
--learning_rate=1e-5 --num_train_epochs 120 --snr_gamma 5 --output_dir="saved/tango_video2audio_vggsound" \
--video_column feature_file \
--audio_column location --text_column captions --checkpointing_steps="best" \
--task video2audio \
--num_warmup_steps 300 \
--embedding_dim 2048 \
--fea_encoder_name "clip-vit-large-patch14" \
> tango_video2audio_vggsound.log 2>&1 

# --input_mol concat \
# --resume_from_checkpoint saved/tango_video2audio_concat/best \
# --unet_model_config="configs/diffusion_model_config_large_2048.json"
# --of True \
# --cavp True \
# --vivit True \
# --jepa True \
# --concat_augment True \
# --resume_from_checkpoint "saved/tango_video2audio_cavp/best" \
# --pe True \
# --lb True \
# --video_column video_path \
# --task video2audio \
