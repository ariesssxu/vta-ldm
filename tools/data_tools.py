import time
import argparse
import json
import logging
import torch
import tqdm
import os
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

import soundfile as sf
import diffusers
import transformers
import torch_tools as torch_tools
from huggingface_hub import snapshot_download
from models import build_pretrained_models, AudioDiffusion
from transformers import SchedulerType, get_scheduler

def extract_video_feature(video_path, feature_path=None):
    files = os.listdir(video_path)
    print("Total files:", len(files))
    if feature_path is not None:
        if not os.path.exists(feature_path):
            os.makedirs(feature_path)
    for video in tqdm(files[:]):
        video_file = os.path.join(video_path, video)
        try:
            video_feature = torch_tools.load_video(video_file, frame_rate=2, size=224)
            # print(video_feature.shape)
            feature_file = os.path.join(feature_path, video.split(".")[0] + ".pt")
            if 16 < video_feature.shape[0] < 24:
                torch.save(video_feature, feature_file)
        except Exception as e:
            print(e)
            print("Error processing:", video_path)
            continue

def extract_audio_feature(video_path, audio_path):
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)
    for video in tqdm(os.listdir(dataset_path)[:]):
        video_path = os.path.join(dataset_path, video)
        audio_file = os.path.join(audio_path, video + ".wav")
        duration = 10
        target_length = int(duration * 102.4)
        try:
            waveform = torch_tools.get_wav_from_video(video_path, target_length * 160, tgt_sr=16000)
            # save audio
            sf.write(audio_file, waveform.squeeze().numpy(), 16000)
        except Exception as e:
            print(e)
            print("Error processing:", video_path)

def load_description(text_path):
    text_dir = {}
    with open(text_path, "r") as f:
        for line in f:
            config = json.loads(line)
            # basename
            video_name = os.path.basename(config["location"])[:-4] + ".mp4"
            text_dir[video_name] = config["captions"]

    return text_dir


def generate_train_config(video_path, feature_path, audio_path, save_dir, text_path="./data/description.txt"):
    configs = []
    # text_dir = load_description(text_path)
    for video in tqdm(os.listdir(video_path)):
        video_path_0 = os.path.join(video_path, video)
        feature_file_0 = os.path.join(feature_path, video + ".pt")
        audio_file_0 = os.path.join(audio_path, video + ".wav")
        # if os.path.exists(feature_file_0) and os.path.exists(audio_file_0) and video in text_dir:
        if os.path.exists(feature_file_0) and os.path.exists(audio_file_0):
            configs.append({
                "video_path": video_path_0,
                "feature_file": feature_file_0,
                "audio_file": audio_file_0,
                # "description": text_dir[video]
                "description": None
            })
    train, test, val = configs[:int(len(configs)*0.96)], configs[int(len(configs)*0.96):int(len(configs)*0.98)], configs[int(len(configs)*0.98):]
    # train, test, val = configs[:2], configs[2:4], configs[4:8]
    print("Train:", len(train), "Test:", len(test), "Val:", len(val))
    with open(os.path.join(save_dir, "train.jsonl"), "w") as f:
        for item in train:
            json.dump(item, f)
            f.write("\n")
    with open(os.path.join(save_dir, "test.jsonl"), "w") as f:
        for item in test:
            json.dump(item, f)
            f.write("\n")
    with open(os.path.join(save_dir, "valid.jsonl"), "w") as f:
        for item in val:
            json.dump(item, f)
            f.write("\n")

if __name__ == "__main__":
    # dataset_path = "youtube_videos/youtube_20240429"
    dataset_path = "vggsound"
    audios = dataset_path + "_wavseg"
    videos = dataset_path + "_mp4seg"
    features = dataset_path + "_feature"
    save_path = "./data/youtube_video"
    text_path = None
    extract_video_feature(videos, features)
    # extract_audio_feature(dataset_path, audio_path)
    # generate_train_config(dataset_path, feature_path, audio_path, save_path, text_path)


    
