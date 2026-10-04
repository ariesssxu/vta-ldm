from .stft import TacotronSTFT

__all__ = ["TacotronSTFT", "wav_to_fbank", "read_wav_file"]


def __getattr__(name):
    """Load TorchAudio-backed helpers only when audio preprocessing needs them."""
    if name in {"wav_to_fbank", "read_wav_file"}:
        from . import tools
        return getattr(tools, name)
    raise AttributeError(name)
