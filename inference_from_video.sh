steps=300
guidance=3
num_samples=1
model="vta-ldm-clip4clip-v-large"
# model="vta_ldm_clip4clip_augment_pe"
# model="vta_ldm_clip4clip_augment_ib"
# model="vta_ldm_clip4clip_ib"
# model="vta_ldm_clip4clip_lb"
# model="vta_ldm_clip4clip_pe"
# model="vta_ldm_clip4clip_text"
# model="vta_ldm_youtube"
# model="vta_ldm_vjepa"
CUDA_VISIBLE_DEVICES=2 python3.10 inference_from_video.py --original_args="ckpt/$model/summary.jsonl" \
--model="ckpt/$model/pytorch_model_2.bin" \
--sample_rate 16000 \
--data_path "data" \
--save_dir outputs/$model \
--num_steps $steps \
--guidance $guidance \
--num_samples $num_samples \
--batch_size 8
