import time
import argparse
import json
import logging
import math
import os
from pathlib import Path
import random
from PIL import Image
import datasets
import numpy as np
import pandas as pd
import wandb
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import librosa
import soundfile as sf
import diffusers
import transformers
import tools.torch_tools as torch_tools
from huggingface_hub import snapshot_download
from models import build_pretrained_models, AudioDiffusion
from transformers import SchedulerType, get_scheduler

import random

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a diffusion model for text/image/video to audio generation task.")
    parser.add_argument(
        "--train_file", type=str, default="data/train_audiocaps.json",
        help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default="data/valid_audiocaps.json",
        help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default="data/test_audiocaps_subset.json",
        help="A csv or a json file containing the test data for generation."
    )
    parser.add_argument(
        "--num_examples", type=int, default=-1,
        help="How many examples to use for training and validation.",
    )
    parser.add_argument(
        "--fea_encoder_name", type=str, default="google/flan-t5-large",
        help="Encoder identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--scheduler_name", type=str, default="stabilityai/stable-diffusion-2-1",
        help="Scheduler identifier.",
    )
    parser.add_argument(
        "--unet_model_name", type=str, default=None,
        help="UNet model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--unet_model_config", type=str, default=None,
        help="UNet model config json path.",
    )
    parser.add_argument(
        "--hf_model", type=str, default=None,
        help="Tango model identifier from huggingface: declare-lab/tango",
    )
    parser.add_argument(
        "--snr_gamma", type=float, default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--freeze_text_encoder", action="store_true", default=False,
        help="Freeze the text encoder model.",
    )
    parser.add_argument(
        "--text_column", type=str, default="captions",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--image_column", type=str, default="img",
        help="The name of the column in the datasets containing the image paths.",
    )
    parser.add_argument(
        "--audio_column", type=str, default="location",
        help="The name of the column in the datasets containing the audio paths.",
    )
    parser.add_argument(
        "--video_column", type=str, default="location",
        help="The name of the column in the datasets containing the video paths.",
    )
    parser.add_argument(
        "--augment", action="store_true", default=False,
        help="Augment training data.",
    )
    parser.add_argument(
        "--uncondition", action="store_true", default=False,
        help="10% uncondition for training.",
    )
    parser.add_argument(
        "--input_mol", default="joint", choices=["concat", "joint", "text", "image", "video"],
    )
    parser.add_argument(
        "--prefix", type=str, default=None,
        help="Add prefix in text prompts.",
    )
    parser.add_argument(
        "--per_device_train_batch_size", type=int, default=2,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, default=2,
        help="Batch size (per device) for the validation dataloader.",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", type=int, default=40,
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps", type=int, default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type", type=SchedulerType, default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0,
        help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9,
        help="The beta1 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999,
        help="The beta2 parameter for the Adam optimizer."
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2,
        help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon", type=float, default=1e-08,
        help="Epsilon value for the Adam optimizer"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps", type=str, default="best",
        help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.",
    )
    parser.add_argument(
        "--save_every", type=int, default=1,
        help="Save model after every how many epochs when checkpointing_steps is set to best."
    )
    parser.add_argument(
        "--resume_from_checkpoint", type=str, default=None,
        help="If the training should continue from a local checkpoint folder.",
    )
    parser.add_argument(
        "--vae_model", type=str, default="audioldm-s-full",
        help="Vae model",
    )
    parser.add_argument(
        "--with_tracking", action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--embedding_dim", type=int, default=1024,
    )
    parser.add_argument(
        "--sample_rate", type=int, default=16000,
        help="Sample rate",
    )
    parser.add_argument(
        "--report_to", type=str, default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--task", choices=['text2audio', 'image2audio', 'video2audio'],
        help="task",
    )
    parser.add_argument(
        "--concat_augment", type=bool, default=False,
        help="Concatenate two videos and audios for training",
    )
    parser.add_argument(
        "--pe", type=bool, default=False,
        help="Use positional embedding or not",
    )
    parser.add_argument(
        "--vivit", type=bool, default=False,
        help="Use VIVIT",
    )
    parser.add_argument(
        "--ib", type=bool, default=False,
        help="Use imagebind or not",
    )
    parser.add_argument(
        "--denseav", type=bool, default=False,
        help="Use denseav or not",
    )
    parser.add_argument(
        "--lb", type=bool, default=False,
        help="Use languagebind or not",
    )
    parser.add_argument(
        "--jepa", type=bool, default=False,
        help="Use jepa or not",
    )
    parser.add_argument(
        "--of", type=bool, default=False,
        help="Use of or not",
    )
    parser.add_argument(
        "--cavp", type=bool, default=False,
        help="Use cavp or not",
    )
    parser.add_argument(
        "--img_pretrained_model_path", type=str, default="None",
        help="image pretrained model path",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "jsonl"], "`validation_file` should be a csv or a json file."

    return args

class Video2AudioDataset(Dataset):
    def __init__(self, dataset, video_column, audio_column, text_column=None, num_examples=-1, concat_augment=False, return_path=False, **kwargs):

        self.videos = list(dataset[video_column])
        self.audios = list(dataset[audio_column])
        self.indices = list(range(len(self.videos)))
        self.return_path = return_path
        self.texts = list(dataset[text_column]) if text_column is not None else [None] * len(self.audios)
        self.concat_augment = concat_augment
        print("*****CONCAT AUGMENT*****") if self.concat_augment else print ("*****NO CONCAT AUGMENT*****")

        self.mapper = {}
        for index, audio, video, text in zip(self.indices, self.audios, self.videos, self.texts):
            self.mapper[index] = [audio, video, text]

        if num_examples != -1:
            self.videos, self.audios, self.texts = self.videos[:num_examples], self.audios[:num_examples], self.texts[:num_examples]
            self.indices = self.indices[:num_examples]

    def __len__(self):
        return len(self.videos)

    def get_num_instances(self):
        return len(self.videos)

    def __getitem__(self, index):
        s1, s2, s3, s4 = self.videos[index], self.audios[index], self.texts[index], self.indices[index]
        
        # pt file is recommended to make full use of the GPU utils
        if s1.endswith(".pt"):
            try:
                s1 = torch.load(s1)
            except Exception as e:
                print(e)
                print("Error processing:", s1)
                s1 = torch.rand(80, 3, 224, 224)
        elif s1.endswith(".mp4"):
            if not self.return_path:
                s1 = torch_tools.load_video(s1, frame_rate=2, size=224)
        s2, sr = sf.read(s2)

        # make sure the audio is 16k
        if sr != 16000:
            s2 = librosa.resample(s2, orig_sr=sr, target_sr=16000)

        # make sure the audio is 10s, otherwise pad it
        s2 = np.pad(s2, (0, 16384*10-len(s2))) if len(s2) < 16384*10 else s2[:16384*10]
        
        if self.concat_augment:
            if random.random() > 0:
                # we only concat part of the traing video
                idx_2 = random.randint(0, len(self.videos)-1)
                s1_1, s2_1, s3_1, s4_1 = self.videos[idx_2], self.audios[idx_2], self.texts[idx_2], self.indices[idx_2]
                s1_1 = torch.load(s1_1)
                s2_1, sr_1 = sf.read(s2_1)
                if sr_1 != 16000:
                    s2_1 = librosa.resample(s2_1, orig_sr=sr_1, target_sr=16000)
                s2_1 = np.pad(s2_1, (0, 16384*10-len(s2_1))) if len(s2_1) < 16384*10 else s2_1[:16384*10]
                cut = random.randint(5, 15)
                s1 = torch.cat([s1[:cut], s1_1[cut:]], dim=0)
                s2 = np.concatenate([s2[:int(sr*cut/2)], s2_1[int(sr_1*cut/2):]], axis=0)
                s3 = s3 + " " + s3_1
        return s1, s2, s3, s4

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [dat[i].tolist() for i in dat]


def main():
    args = parse_args()
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    datasets.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory creation and wandb tracking
    if accelerator.is_main_process:
        if args.output_dir is None or args.output_dir == "":
            args.output_dir = "saved/" + str(int(time.time()))
            
            if not os.path.exists("saved"):
                os.makedirs("saved")
                
            os.makedirs(args.output_dir, exist_ok=True)
            
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        os.makedirs("{}/{}".format(args.output_dir, "outputs"), exist_ok=True)
        with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
            f.write(json.dumps(dict(vars(args))) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

        #wandb.init(project="Text to Audio Diffusion")

    accelerator.wait_for_everyone()

    # Get the datasets
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    if args.test_file is not None:
        data_files["test"] = args.test_file
    else:
        if args.validation_file is not None:
            data_files["test"] = args.validation_file

    extension = args.train_file.split(".")[-1]
    raw_datasets = load_dataset(extension, data_files=data_files)
    text_column, image_column, video_column, audio_column = args.text_column, args.image_column, args.video_column, args.audio_column
    text_column = None if text_column == "None" else text_column

    # Initialize models
    pretrained_model_name = args.vae_model
    sample_rate = args.sample_rate
    vae, stft = build_pretrained_models(pretrained_model_name)
    vae.eval()
    stft.eval()

    model_class = AudioDiffusion
    # if args.ib:
    #     print("*****USING MODEL IMAGEBIND*****")
    #     from models_imagebind import AudioDiffusion_IB
    #     model_class = AudioDiffusion if not args.ib else AudioDiffusion_IB
    # elif args.lb:
    #     print("*****USING MODEL LANGUAGEBIND*****")
    #     from models_languagebind import AudioDiffusion_LB
    #     model_class = AudioDiffusion_LB
    # elif args.jepa:
    #     print("*****USING MODEL JEPA*****")
    #     from models_vjepa import AudioDiffusion_JEPA
    #     model_class = AudioDiffusion_JEPA
    # elif args.cavp:
    #     print("*****USING MODEL CAVP*****")
    #     from models_cavp import AudioDiffusion_CAVP
    #     model_class = AudioDiffusion_CAVP
    # elif args.vivit:
    #     print("*****USING MODEL VIVIT*****")
    #     from models_vivit import AudioDiffusion_VIVIT
    #     model_class = AudioDiffusion_VIVIT
    # elif args.denseav:
    #     print("*****USING MODEL DENSEAV*****")
    #     from models_denseav import AudioDiffusion_DenseAV
    #     model_class = AudioDiffusion_DenseAV
    # elif args.of:
    #     print("*****USING MODEL OF*****")
    #     # from models_of import AudioDiffusion_OF
    #     # model_class = AudioDiffusion_OF

    from models import AudioDiffusion
        
    model = model_class(
        args.fea_encoder_name, 
        args.scheduler_name, 
        args.unet_model_name, 
        args.unet_model_config, 
        args.snr_gamma, 
        args.freeze_text_encoder, 
        args.uncondition, 
        args.img_pretrained_model_path, 
        args.task,
        args.embedding_dim,
        args.pe
    )

    if args.hf_model:
        #hf_model_path = snapshot_download(repo_id=args.hf_model)
        hf_model_path = args.hf_model
        model.load_state_dict(torch.load("{}".format(hf_model_path), map_location="cpu"))
        accelerator.print("Successfully loaded checkpoint from:", args.hf_model)
        
    with accelerator.main_process_first():
        return_path = False if not args.cavp else True
        train_dataset = Video2AudioDataset(raw_datasets["train"], video_column, audio_column, text_column, args.num_examples, args.concat_augment, return_path)
        eval_dataset = Video2AudioDataset(raw_datasets["validation"], video_column, audio_column, text_column, args.num_examples, args.concat_augment, return_path)
        test_dataset = Video2AudioDataset(raw_datasets["test"], video_column, audio_column, text_column, args.num_examples, args.concat_augment, return_path)
        accelerator.print("Num instances in train: {}, validation: {}, test: {}".format(train_dataset.get_num_instances(), eval_dataset.get_num_instances(), test_dataset.get_num_instances()))

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size, collate_fn=train_dataset.collate_fn, num_workers=16)
    eval_dataloader = DataLoader(eval_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size, collate_fn=eval_dataset.collate_fn, num_workers=16)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=args.per_device_eval_batch_size, collate_fn=test_dataset.collate_fn, num_workers=16)

    # Optimizer
    if args.task == "video2audio":
        if args.ib:
            for param in model.ib_model.parameters():
                param.requires_grad = False
        elif args.lb:
            for param in model.lb_model.parameters():
                param.requires_grad = False
        elif args.jepa:
            for param in model.vjepa_model.parameters():
                param.requires_grad = False
        elif args.cavp:
            for param in model.cavp.parameters():
                param.requires_grad = False
        elif args.vivit:
            for param in model.vivit.parameters():
                param.requires_grad = False
        elif args.denseav:
            for param in model.denseav_model.parameters():
                param.requires_grad = False
        # elif args.of:
        #     print("")
        #     if args.input_mol == "concat" or args.input_mol == "joint":
        #         for param in model.vid_fea_extractor.clip_text.parameters():
        #             param.requires_grad = False
        else:
            for param in model.vid_fea_extractor.parameters():
                param.requires_grad = False
            for param in model.vid_fea_extractor.linear_layer.parameters():
                param.requires_grad = True
        
    elif args.task == "text2audio":
        # take it simple
        for param in model.CLIP_model.parameters():
            param.requires_grad = False
    if args.unet_model_config:
        optimizer_parameters = model.unet.parameters()
        accelerator.print("Optimizing UNet parameters.")
    else:
        optimizer_parameters = list(model.unet.parameters()) + list(model.group_in.parameters()) + list(model.group_out.parameters())
        accelerator.print("Optimizing UNet and channel transformer parameters.")

    num_trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accelerator.print("Num trainable parameters: {}".format(num_trainable_parameters))

    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    vae, stft, model, optimizer, lr_scheduler = accelerator.prepare(
        vae, stft, model, optimizer, lr_scheduler
    )

    train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader, test_dataloader
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        accelerator.init_trackers("text_to_audio_diffusion", experiment_config)

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)

    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            accelerator.load_state(args.resume_from_checkpoint, strict=False)
            # path = os.path.basename(args.resume_from_checkpoint)
            accelerator.print(f"Resumed from local checkpoint: {args.resume_from_checkpoint}")
        else:
            # Get the most recent checkpoint
            dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            # path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            
    # Duration of the audio clips in seconds
    duration, best_loss = 10, np.inf
    input_mol = args.input_mol
    
    total_video_epochs = 0
    total_text_epochs = 0
    
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        for step, batch in enumerate(train_dataloader):
            # random sample input modalities
            if args.input_mol == "joint":
                input_mol = random.choice(["text", "video"])
                total_video_epochs += 1 if input_mol == "video" else 0
                total_text_epochs += 1 if input_mol == "text" else 0
                if accelerator.is_main_process and step  % 100 == 0:
                    accelerator.print("Total video epochs: {}, Total text epochs: {}".format(total_video_epochs, total_text_epochs))
                # input_mol = "text"
            elif args.input_mol == "concat":
                input_mol = "concat"
            elif args.input_mol == "video":
                input_mol = "video"
            elif args.input_mol == "text":
                input_mol = "text"

            with accelerator.accumulate(model):
                device = model.device
                videos, audios, texts, _ = batch
                duration = 10
                target_length = int(duration * 102.4)

                with torch.no_grad():
                    unwrapped_vae = accelerator.unwrap_model(vae)
                    audios = torch.from_numpy(np.array(audios)).float()
                    mel, _, waveform = torch_tools.wav_in_video_to_fbank(audios, target_length, sample_rate, stft, waveform=True)
                    mel = mel.unsqueeze(1).to(device)
                    true_latent = unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel))

                if input_mol == "text":
                    loss = model(true_latent, text=texts, validation_mode=False, device=device)
                elif input_mol == "video":
                    loss = model(true_latent, video=videos, validation_mode=False, device=device)
                elif input_mol == "concat":
                    loss = model(true_latent, video=videos, text=texts, validation_mode=False, device=device)

                total_loss += loss.detach().float()
                accelerator.backward(loss)
                # if accelerator.is_main_process:
                #     print(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)

            if completed_steps >= args.max_train_steps:
                break
            
        model.eval()
        model.uncondition = False

        eval_progress_bar = tqdm(range(len(eval_dataloader)), disable=not accelerator.is_local_main_process)
        input_mol = "video" if args.input_mol in ["video", "joint"] else input_mol
        for step, batch in enumerate(eval_dataloader):
            with accelerator.accumulate(model) and torch.no_grad():
                device = model.device
                videos, audios, texts, _ = batch
                target_length = int(duration * 102.4)
                unwrapped_vae = accelerator.unwrap_model(vae)
                audios = torch.from_numpy(np.array(audios)).float()
                mel, _, waveform = torch_tools.wav_in_video_to_fbank(audios, target_length, sample_rate, stft, waveform=True)
                mel = mel.unsqueeze(1).to(device)
                true_latent = unwrapped_vae.get_first_stage_encoding(unwrapped_vae.encode_first_stage(mel))
                true_latent = true_latent.to(device)

                if input_mol == "text":
                    val_loss = model(true_latent, text=texts, validation_mode=False, device=device)
                elif input_mol == "video":
                    val_loss = model(true_latent, video=videos, validation_mode=False, device=device)
                elif input_mol == "concat":
                    val_loss = model(true_latent, video=videos, text=texts, validation_mode=False, device=device)

                total_val_loss += val_loss.detach().float()
                eval_progress_bar.update(1)

        model.uncondition = args.uncondition

        if accelerator.is_main_process:    
            result = {}
            result["epoch"] = epoch+1,
            result["step"] = completed_steps
            result["train_loss"] = round(total_loss.item()/len(train_dataloader), 4)
            result["val_loss"] = round(total_val_loss.item()/len(eval_dataloader), 4)

            #wandb.log(result)

            result_string = "Epoch: {}, Loss Train: {}, Val: {}, lr: {} \n".format(epoch, result["train_loss"], result["val_loss"], optimizer.param_groups[0]["lr"])
            
            accelerator.print(result_string)

            with open("{}/summary.jsonl".format(args.output_dir), "a") as f:
                f.write(json.dumps(result) + "\n\n")

            logger.info(result)

            if result["val_loss"] < best_loss:
                best_loss = result["val_loss"]
                save_checkpoint = True
            else:
                save_checkpoint = False

        if args.with_tracking:
            accelerator.log(result, step=completed_steps)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process and args.checkpointing_steps == "best":
            if save_checkpoint:
                accelerator.save_state("{}/{}".format(args.output_dir, "best"))
                
            if (epoch + 1) % args.save_every == 0:
                accelerator.save_state("{}/{}".format(args.output_dir, "epoch_" + str(epoch+1)))

        if accelerator.is_main_process and args.checkpointing_steps == "epoch":
            accelerator.save_state("{}/{}".format(args.output_dir, "epoch_" + str(epoch+1)))
            
            
if __name__ == "__main__":
    main()
