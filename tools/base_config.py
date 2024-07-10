
import os

def default_vae_config():    
    basic_config = {
        "model": {
            "params": {
                "first_stage_config": {
                    "base_learning_rate": 4.5e-05,
                    "target": "audioldm.variational_autoencoder.autoencoder.AutoencoderKL",
                    "params": {
                        "monitor": "val/rec_loss",
                        "image_key": "fbank",
                        "subband": 1,
                        "embed_dim": 8,
                        "time_shuffle": 1,
                        "ddconfig": {
                            "double_z": True,
                            "z_channels": 8,
                            "resolution": 256,
                            "downsample_time": False,
                            "in_channels": 1,
                            "out_ch": 1,
                            "ch": 128,
                            "ch_mult": [1, 2, 4],
                            "num_res_blocks": 2,
                            "attn_resolutions": [],
                            "dropout": 0.0,
                        },
                    },
                },
            },
        },
    }
    
        
    return basic_config

def default_stft_config():    
    
    basic_config = {
        "preprocessing_16k": {
            "audio": {"sampling_rate": 16000, "max_wav_value": 32768},
            "stft": {"filter_length": 1024, "hop_length": 160, "win_length": 1024},
            "mel": {
                "n_mel_channels": 64,
                "mel_fmin": 0,
                "mel_fmax": 8000,
                "freqm": 0,
                "timem": 0,
                "blur": False,
                "mean": -4.63,
                "std": 2.74,
                "target_length": 1024,
            },
        },
        "preprocessing_24k": {
            "audio": {"sampling_rate": 24000, "max_wav_value": 32768},
            "stft": {"filter_length": 2048, "hop_length": 240, "win_length": 2048},
            "mel": {
                "n_mel_channels": 64,
                "mel_fmin": 0,
                "mel_fmax": 12000,
                "target_length": 1024,
            },
        },
        "preprocessing_32k": {
            "audio": {"sampling_rate": 32000, "max_wav_value": 32768},
            "stft": {"filter_length": 2048, "hop_length": 320, "win_length": 2048},
            "mel": {
                "n_mel_channels": 64,
                "mel_fmin": 0,
                "mel_fmax": 16000,
                "target_length": 1024,
            },
        },
        "preprocessing_48k": {
            "audio": {"sampling_rate": 48000, "max_wav_value": 32768, "duration": 10.00},
            "stft": {"filter_length": 2048, "hop_length": 480, "win_length": 2048},
            "mel": {
                "n_mel_channels": 64,
                "mel_fmin": 20,
                "mel_fmax": 24000
            }
        },
    }
    
    return basic_config

def get_metadata():
    return {
        "audioldm-s-full": {
            "path": os.path.join(
                CACHE_DIR,
                "audioldm-s-full.ckpt",
            ),
            "url": "https://zenodo.org/record/7600541/files/audioldm-s-full?download=1",
        },
        "audioldm-l-full": {
            "path": os.path.join(
                CACHE_DIR,
                "audioldm-l-full.ckpt",
            ),
            "url": "https://zenodo.org/record/7698295/files/audioldm-full-l.ckpt?download=1",
        },
        "audioldm-s-full-v2": {
            "path": os.path.join(
                CACHE_DIR,
                "audioldm-s-full-v2.ckpt",
            ),
            "url": "https://zenodo.org/record/7698295/files/audioldm-full-s-v2.ckpt?download=1",
        },
        "audioldm-m-text-ft": {
            "path": os.path.join(
                CACHE_DIR,
                "audioldm-m-text-ft.ckpt",
            ),
            "url": "https://zenodo.org/record/7813012/files/audioldm-m-text-ft.ckpt?download=1",
        },
        "audioldm-s-text-ft": {
            "path": os.path.join(
                CACHE_DIR,
                "audioldm-s-text-ft.ckpt",
            ),
            "url": "https://zenodo.org/record/7813012/files/audioldm-s-text-ft.ckpt?download=1",
        },
        "audioldm-m-full": {
            "path": os.path.join(
                CACHE_DIR,
                "audioldm-m-full.ckpt",
            ),
            "url": "https://zenodo.org/record/7813012/files/audioldm-m-full.ckpt?download=1",
        },
    }

