import torch
import torchaudio
import random
import itertools
import numpy as np
from tools.mix import mix
from PIL import Image
import cv2
from moviepy.editor import VideoFileClip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode, RandomResizedCrop

def normalize_wav(waveform):
    waveform = waveform - torch.mean(waveform)
    waveform = waveform / (torch.max(torch.abs(waveform)) + 1e-8)
    return waveform * 0.5

def sinusoidal_positional_embedding(token_sequence_size, token_embedding_dim, n=10000.0):

    if token_embedding_dim % 2 != 0:
        raise ValueError("Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={:d})".format(token_embedding_dim))

    T = token_sequence_size
    d = token_embedding_dim #d_model=head_num*d_k, not d_q, d_k, d_v

    positions = torch.arange(0, T).unsqueeze_(1)
    embeddings = torch.zeros(T, d)

    denominators = torch.pow(n, 2*torch.arange(0, d//2)/d) # 10000^(2i/d_model), i is the index of embedding
    embeddings[:, 0::2] = torch.sin(positions/denominators) # sin(pos/10000^(2i/d_model))
    embeddings[:, 1::2] = torch.cos(positions/denominators) # cos(pos/10000^(2i/d_model))

    return embeddings

def pad_wav(waveform, segment_length):
    waveform_length = len(waveform)
    
    if segment_length is None or waveform_length == segment_length:
        return waveform
    elif waveform_length > segment_length:
        return waveform[:segment_length]
    else:
        pad_wav = torch.zeros(segment_length - waveform_length).to(waveform.device)
        waveform = torch.cat([waveform, pad_wav])
        return waveform
    
    
def _pad_spec(fbank, target_length=1000):
    batch, n_frames, channels = fbank.shape
    p = target_length - n_frames
    if p > 0:
        pad = torch.zeros(batch, p, channels).to(fbank.device)
        fbank = torch.cat([fbank, pad], 1)
    elif p < 0:
        fbank = fbank[:, :target_length, :]

    if channels % 2 != 0:
        fbank = fbank[:, :, :-1]

    return fbank


def read_wav_file(filename, segment_length, tgt_sr=48000):
    waveform, sr = torchaudio.load(filename)  # Faster!!!
    if sr != tgt_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=tgt_sr)[0]
    else:
        waveform = waveform.squeeze()
    try:
        waveform = normalize_wav(waveform)
    except:
        print ("Exception normalizing:", filename)
        waveform = torch.ones(tgt_sr * 10)
    waveform = pad_wav(waveform, segment_length).unsqueeze(0)
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform
    return waveform


def get_mel_from_wav(audio, _stft):
    audio1 = torch.nan_to_num(torch.clip(audio, -1, 1))
    audio2 = torch.autograd.Variable(audio1, requires_grad=False)
    melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio2)
    return melspec, log_magnitudes_stft, energy

def wav_to_fbank(paths, target_length=1000, sample_rate=16000, fn_STFT=None):
    assert fn_STFT is not None
    if sample_rate == 16000:
        hop_size = 160
    elif sample_rate == 24000:
        hop_size = 240
    elif sample_rate == 32000:
        hop_size = 320
    elif sample_rate == 48000:
        hop_size = 480
    else:
        raise ValueError(f"sample_rate wrong.") 

    #print("target_length", target_length, hop_size)
    #print("target_length", target_length, sample_rate, fn_STFT)
    #for name, param in fn_STFT.named_parameters():
    #    print(name, param.data)
    waveform = torch.cat([read_wav_file(path, target_length * hop_size, tgt_sr=sample_rate) for path in paths], 0)  # hop size is 160
    #print("waveform", waveform.size())

    #np.set_printoptions(threshold=np.inf)
    #print("waveform", waveform)
    #f_out = open(paths[0].split("/")[-1]+".scp",'w')
    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    #print("fbank", fbank)
    fbank = fbank.transpose(1, 2)
    log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )

    #f_out.write(paths[0]+ "\n" + str(waveform.cpu().numpy())+"\n")
    #f_out.write("audio1"+ "\n" + str(audio1.cpu().numpy())+"\n")
    #f_out.write("audio2"+ "\n" + str(audio2.cpu().numpy())+"\n")
    #f_out.write("fbank" + "\n" + str(fbank.cpu().numpy())+"\n")
    #print(fbank2)
    return fbank, log_magnitudes_stft, waveform

def get_wav_from_video(video_path, segment_length, tgt_sr=48000):
    video = VideoFileClip(video_path)
    audio = video.audio
    sr = audio.fps
    audio_data = audio.to_soundarray() # 441882 * 2 双通道
    waveform = torch.mean(torch.tensor(audio_data, dtype=torch.float), dim=1).unsqueeze(0)  # 变成单通道
    if sr != tgt_sr:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=tgt_sr)[0]
    else:
        waveform = waveform.squeeze()
    try:
        waveform = normalize_wav(waveform)
    except:
        print ("Exception normalizing:", video_path)
        waveform = torch.ones(tgt_sr * 10)
    waveform = pad_wav(waveform, segment_length).unsqueeze(0)
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform
    return waveform

def get_wavs_from_videos(video_paths, segment_length, tgt_sr=48000):
    wavs = []
    for video_path in video_paths:
        waveform = get_wav_from_video(video_path, segment_length, tgt_sr)
        wavs.append(waveform)
    wavs = torch.cat(wavs, 0)
    return wavs

def wav_in_video_to_fbank(input, target_length=1000, sample_rate=16000, fn_STFT=None, waveform=False):
    assert fn_STFT is not None
    if sample_rate == 16000:
        hop_size = 160
    elif sample_rate == 24000:
        hop_size = 240
    elif sample_rate == 32000:
        hop_size = 320
    elif sample_rate == 48000:
        hop_size = 480
    else:
        raise ValueError(f"sample_rate wrong.") 

    if not waveform:
        paths = input
        waveform = get_wavs_from_videos(paths, target_length * hop_size, tgt_sr=sample_rate) # hop size is 160
    else:
        waveform = input
    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    fbank = fbank.transpose(1, 2)
    log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )
    return fbank, log_magnitudes_stft, waveform


def uncapitalize(s):
    if s:
        return s[:1].lower() + s[1:]
    else:
        return ""

    
def mix_wavs_and_captions(path1, path2, caption1, caption2, target_length=1000, sample_rate=16000):

    if sample_rate == 16000:
        hop_size = 160
    elif sample_rate == 24000:
        hop_size = 240
    elif sample_rate == 32000:
        hop_size = 320
    elif sample_rate == 48000:
        hop_size = 480
    else:
        raise ValueError(f"sample_rate wrong.")

    sound1 = read_wav_file(path1, target_length * hop_size)[0].numpy()
    #print("sound1", target_length, sound1.size)
    sound2 = read_wav_file(path2, target_length * hop_size)[0].numpy()
    mixed_sound = mix(sound1, sound2, 0.5, sample_rate).reshape(1, -1)
    #print("mixed_sound", mixed_sound.size)
    mixed_caption = "{} and {}".format(caption1, uncapitalize(caption2))
    return mixed_sound, mixed_caption


def augment(paths, texts, num_items=4, target_length=1000, sample_rate=16000):
    mixed_sounds, mixed_captions = [], []
    combinations = list(itertools.combinations(list(range(len(texts))), 2))
    random.shuffle(combinations)
    if len(combinations) < num_items:
        selected_combinations = combinations
    else:
        selected_combinations = combinations[:num_items]
        
    for (i, j) in selected_combinations:
        new_sound, new_caption = mix_wavs_and_captions(paths[i], paths[j], texts[i], texts[j], target_length, sample_rate)
        mixed_sounds.append(new_sound)
        mixed_captions.append(new_caption)
        
    waveform = torch.tensor(np.concatenate(mixed_sounds, 0))
    waveform = waveform / torch.max(torch.abs(waveform))
    waveform = 0.5 * waveform
    
    return waveform, mixed_captions


def augment_wav_to_fbank(paths, texts, num_items=4, target_length=1000, sample_rate=16000, fn_STFT=None):
    assert fn_STFT is not None
    
    waveform, captions = augment(paths, texts, target_length = target_length, sample_rate=sample_rate)
    fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)
    fbank = fbank.transpose(1, 2)
    log_magnitudes_stft = log_magnitudes_stft.transpose(1, 2)

    fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
        log_magnitudes_stft, target_length
    )

    return fbank, log_magnitudes_stft, waveform, captions


def load_image(impaths, crop_size=384):
    imgs = []
    RGB_mean = [0.485, 0.456, 0.406]
    RGB_std = [0.229, 0.224, 0.225]
    image_resize_and_crop = Compose([RandomResizedCrop(crop_size), ToTensor()])
    image_normalize = Normalize(mean=RGB_mean, std=RGB_std)
    for impath in impaths:
        img = Image.open(impath).convert('RGB')
        img = image_resize_and_crop(img)
        img = image_normalize(img)
        imgs.append(img)
    imgs = torch.stack(imgs)

    return imgs

def load_video(video_path, frame_rate=1.0, size=224):
    def preprocess(size, n_px):
        return Compose([
            Resize(size, interpolation=InterpolationMode.BICUBIC),            
            CenterCrop(size),
            lambda image: image.convert("RGB"),
            ToTensor(),
            # Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])(n_px)
    videos = []
    # for video_path in video_paths:
    # cap = cv2.VideoCapture(video_path)
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps < 1:
        images = np.zeros([3, size, size], dtype=np.float32) 
        print("ERROR: problem reading video file: ", video_path)
    else:
        total_duration = (frameCount + fps - 1) // fps
        start_sec, end_sec = 0, total_duration
        interval = fps / frame_rate
        frames_idx = np.floor(np.arange(start_sec*fps, end_sec*fps, interval))
        ret = True     
        images = np.zeros([len(frames_idx), 3, size, size], dtype=np.float32)
            
        for i, idx in enumerate(frames_idx):
            cap.set(cv2.CAP_PROP_POS_FRAMES , idx)
            ret, frame = cap.read()    
            if not ret: break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)             
            last_frame = i
            images[i,:,:,:] = preprocess(size, Image.fromarray(frame).convert("RGB"))
            
        images = images[:last_frame+1]
        cap.release()
    return torch.tensor(images)