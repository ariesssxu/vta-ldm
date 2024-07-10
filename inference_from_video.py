import os
import copy
import json
import time
import torch
import argparse
from PIL import Image
import numpy as np
import soundfile as sf
#import wandb
from tqdm import tqdm
from diffusers import DDPMScheduler
from models import build_pretrained_models, AudioDiffusion
from transformers import AutoProcessor, ClapModel
import torchaudio
import tools.torch_tools as torch_tools
from datasets import load_dataset

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def parse_args():
    parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
    parser.add_argument(
        "--original_args", type=str, default=None,
        help="Path for summary jsonl file saved during training."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path for saved model bin file."
    )
    parser.add_argument(
        "--vae_model", type=str, default="audioldm-s-full",
        help="Path for saved model bin file."
    )
    parser.add_argument(
        "--num_steps", type=int, default=200,
        help="How many denoising steps for generation.",
    )
    parser.add_argument(
        "--guidance", type=float, default=3,
        help="Guidance scale for classifier free guidance."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size for generation.",
    )
    parser.add_argument(
        "--num_samples", type=int, default=1,
        help="How many samples per prompt.",
    )
    parser.add_argument(
        "--num_test_instances", type=int, default=-1,
        help="How many test instances to evaluate.",
    )
    parser.add_argument(
        "--sample_rate", type=int, default=-1,
        help="How many test instances to evaluate.",
    )
    parser.add_argument(
        "--save_dir", type=str, default="./outputs/tmp",
        help="output save dir"
    )
    parser.add_argument(
        "--data_path", type=str, default="data/video_processed/video_gt_augment",
        help="inference data path"
    )
    
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    
    train_args = dotdict(json.loads(open(args.original_args).readlines()[0]))
    if "hf_model" not in train_args:
        train_args["hf_model"] = None
    
    # Load Models #
    name = train_args.vae_model
    vae, stft = build_pretrained_models(name)
    vae, stft = vae.cuda(), stft.cuda()
    model_class = AudioDiffusion
    if train_args.ib:
        print("*****USING MODEL IMAGEBIND*****")
        from models_imagebind import AudioDiffusion_IB
        model_class = AudioDiffusion if not train_args.ib else AudioDiffusion_IB
    elif train_args.lb:
        print("*****USING MODEL LANGUAGEBIND*****")
        from models_languagebind import AudioDiffusion_LB
        model_class = AudioDiffusion_LB
    elif train_args.jepa:
        print("*****USING MODEL JEPA*****")
        from models_vjepa import AudioDiffusion_JEPA
        model_class = AudioDiffusion_JEPA

    model = model_class(
        train_args.fea_encoder_name, 
        train_args.scheduler_name, 
        train_args.unet_model_name, 
        train_args.unet_model_config, 
        train_args.snr_gamma, 
        train_args.freeze_text_encoder, 
        train_args.uncondition, 
        train_args.img_pretrained_model_path, 
        train_args.task,
        train_args.embedding_dim,
        train_args.pe
    )
    
    model.eval()

    # Load Trained Weight #
    device = torch.device("cuda:0") #vae.device()
    if args.model.endswith(".pt") or args.model.endswith(".bin"):
        model.load_state_dict(torch.load(args.model), strict=False)
    else:
        from safetensors.torch import load_model
        load_model(model, args.model, strict=False)
        
    model.to(device)
    
    scheduler = DDPMScheduler.from_pretrained(train_args.scheduler_name, subfolder="scheduler")
    sample_rate = args.sample_rate
    #evaluator = EvaluationHelper(16000, "cuda:0")
    

    def audio_text_matching(waveforms, text, sample_freq=24000, max_len_in_seconds=10):
        new_freq = 48000
        resampled = []
        
        for wav in waveforms:
            x = torchaudio.functional.resample(torch.tensor(wav, dtype=torch.float).reshape(1, -1), orig_freq=sample_freq, new_freq=new_freq)[0].numpy()
            resampled.append(x[:new_freq*max_len_in_seconds])

        inputs = clap_processor(text=text, audios=resampled, return_tensors="pt", padding=True, sampling_rate=48000)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = clap(**inputs)

        logits_per_audio = outputs.logits_per_audio
        ranks = torch.argsort(logits_per_audio.flatten(), descending=True).cpu().numpy()
        return ranks
    
    # Load Data #
    if train_args.prefix:
        prefix = train_args.prefix
    else:
        prefix = ""

    # data_path = "data/video_test/"
    data_path = args.data_path
    wavname = [f"{name.split('.')[0]}.wav" for name in os.listdir(data_path)]
    video_features = []
    for video_file in os.listdir(data_path):
        video_path = os.path.join(data_path, video_file)
        video_feature = torch_tools.load_video(video_path, frame_rate=2, size=224)
        print(video_feature.shape)
        video_features.append(video_feature)
    
    # Generate #
    num_steps, guidance, batch_size, num_samples = args.num_steps, args.guidance, args.batch_size, args.num_samples
    all_outputs = []
        
    for k in tqdm(range(0, len(wavname), batch_size)):
        
        with torch.no_grad():
            # if train_args.task == 'image2audio':
            #     prompt = text_prompts[k: k+batch_size]
            #     imgs = []
            #     for img_path in prompt:
            #         img = Image.open(img_path)
            #         imgs.append(np.array(img))
            #     prompt = imgs
            # elif train_args.task == 'video2audio':
            prompt = video_features[k: k+batch_size]

            latents = model.inference(scheduler, None, prompt, None, num_steps, guidance, num_samples, disable_progress=True, device=device)
            mel = vae.decode_first_stage(latents)
            wave = vae.decode_to_waveform(mel)
            
            all_outputs += [item for item in wave]
            
    # Save #
    exp_id = str(int(time.time()))
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    
    if num_samples == 1:
        output_dir = "{}/{}_{}_steps_{}_guidance_{}_sampleRate_{}_augment".format(args.save_dir, exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, sample_rate)
        os.makedirs(output_dir, exist_ok=True)
        for j, wav in enumerate(all_outputs):
            sf.write("{}/{}".format(output_dir, wavname[j]), wav, samplerate=sample_rate)
            
    else:
        for i in range(num_samples):
            output_dir = "{}/{}_{}_steps_{}_guidance_{}_sampleRate_{}/rank_{}".format(args.save_dir, exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, sample_rate, i+1)
            os.makedirs(output_dir, exist_ok=True)
        
        groups = list(chunks(all_outputs, num_samples))
        for k in tqdm(range(len(groups))):
            wavs_for_text = groups[k]
            rank = audio_text_matching(wavs_for_text, text_prompts[k])
            ranked_wavs_for_text = [wavs_for_text[r] for r in rank]
            
            for i, wav in enumerate(ranked_wavs_for_text):
                output_dir = "{}/{}_{}_steps_{}_guidance_{}_sampleRate_{}/rank_{}".format(args.save_dir, exp_id, "_".join(args.model.split("/")[1:-1]), num_steps, guidance, sample_rate, i+1)
                sf.write("{}/{}".format(output_dir, wavname[k]), wav, samplerate=sample_rate)
            
if __name__ == "__main__":
    main()
