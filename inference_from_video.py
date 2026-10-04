import os
# Keep third-party caches writable in restricted/containerized environments.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/vta-ldm-numba")
os.environ.setdefault("HF_HOME", "/tmp/vta-ldm-huggingface")

import json
import torch
import argparse
import random
from pathlib import Path
import numpy as np
import soundfile as sf
from tqdm import tqdm
from models import build_pretrained_models, AudioDiffusion, load_scheduler
from tools.video_tools import load_video

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Inference for text to audio generation task.")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional JSON config. Explicit CLI arguments override its values."
    )
    parser.add_argument(
        "--original_args", type=str, default=None,
        help="Path for summary jsonl file saved during training."
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path for saved model bin file."
    )
    parser.add_argument(
        "--scheduler_config", type=str, default=None,
        help="Local scheduler JSON; avoids fetching the gated SD 2.1 repository."
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
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda"], default="auto",
        help="Execution device. 'auto' selects CUDA when available."
    )

    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.config is not None:
        with preliminary.config.open(encoding="utf-8") as handle:
            defaults = json.load(handle)
        known = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - known)
        if unknown:
            parser.error("unknown config keys: {}".format(", ".join(unknown)))
        parser.set_defaults(**defaults)

    args = parser.parse_args(argv)
    if args.num_steps < 1 or args.batch_size < 1:
        parser.error("num_steps and batch_size must be positive")
    if args.guidance <= 1:
        parser.error("the released video model requires guidance > 1")
    if args.num_samples != 1:
        parser.error("minimal inference currently requires num_samples=1")
    if args.num_test_instances == 0 or args.num_test_instances < -1:
        parser.error("num_test_instances must be -1 or a positive integer")

    return args

def main():
    args = parse_args()
    if args.original_args is None or args.model is None:
        raise ValueError("--original_args and --model are required")
    for path, label in ((args.original_args, "training config"), (args.model, "model checkpoint")):
        if not os.path.isfile(path):
            raise FileNotFoundError("{} not found: {}".format(label, path))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    with open(args.original_args, encoding="utf-8") as handle:
        first_line = handle.readline()
    train_args = dotdict(json.loads(first_line))
    if args.scheduler_config:
        train_args.scheduler_name = args.scheduler_config
    if "hf_model" not in train_args:
        train_args["hf_model"] = None
    
    # Load Models #
    name = train_args.vae_model
    vae, stft = build_pretrained_models(name)
    vae, stft = vae.to(device), stft.to(device)
    model_class = AudioDiffusion
    variants = [name for name in ("ib", "lb", "jepa", "cavp", "vivit", "denseav", "of")
                if train_args.get(name, False)]
    if variants:
        raise ValueError(
            "minimal inference supports the released base model only; "
            "disabled variants: {}".format(", ".join(variants))
        )

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
    if args.model.endswith(".pt") or args.model.endswith(".bin"):
        model.load_state_dict(torch.load(args.model, map_location="cpu"), strict=False)
    else:
        from safetensors.torch import load_model
        load_model(model, args.model, strict=False)
        
    model.to(device)
    
    scheduler = load_scheduler(train_args.scheduler_name)
    sample_rate = args.sample_rate
    #evaluator = EvaluationHelper(16000, "cuda:0")
    

    # Load a deterministic, bounded list of videos.
    data_path = args.data_path
    supported = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    video_files = sorted(
        name for name in os.listdir(data_path)
        if os.path.splitext(name)[1].lower() in supported
    )
    if args.num_test_instances > 0:
        video_files = video_files[:args.num_test_instances]
    if not video_files:
        raise ValueError("no supported videos found in {}".format(data_path))
    wavname = [f"{os.path.splitext(name)[0]}.wav" for name in video_files]
    video_features = []
    for video_file in video_files:
        video_path = os.path.join(data_path, video_file)
        video_feature = load_video(video_path, frame_rate=2, size=224)
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
    os.makedirs(args.save_dir, exist_ok=True)
    
    if num_samples == 1:
        output_dir = "{}/steps_{}_guidance_{}_seed_{}".format(
            args.save_dir, num_steps, guidance, args.seed
        )
        os.makedirs(output_dir, exist_ok=True)
        for j, wav in enumerate(all_outputs):
            sf.write("{}/{}".format(output_dir, wavname[j]), wav, samplerate=sample_rate)
            
if __name__ == "__main__":
    main()
