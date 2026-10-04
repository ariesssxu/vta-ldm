"""Video-only preprocessing without the training stack's audio dependencies."""

from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Resize, ToTensor


def load_video(video_path, frame_rate=1.0, size=224):
    if frame_rate <= 0:
        raise ValueError("frame_rate must be positive")
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    transform = Compose([
        Resize(size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(size),
        ToTensor(),
    ])
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if source_fps <= 0 or frame_count <= 0:
            raise ValueError(f"cannot read video metadata: {path}")
        indices = np.arange(0, frame_count, source_fps / frame_rate).astype(int)
        frames = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(Image.fromarray(rgb)))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"no frames decoded from video: {path}")
    return torch.stack(frames)
