"""AudioLDM components with lazy imports.

Importing a small submodule (for example the STFT used by VTA-LDM) must not
eagerly initialize the full text-to-audio pipeline and all optional packages.
"""

_UTILITIES = {"seed_everything", "save_wave", "get_time", "get_duration"}
_PIPELINE = {"text_to_audio", "style_transfer", "build_model", "round_up_duration"}
__all__ = ["LatentDiffusion", *_UTILITIES, *_PIPELINE]


def __getattr__(name):
    if name == "LatentDiffusion":
        from .ldm import LatentDiffusion
        return LatentDiffusion
    if name in _UTILITIES:
        from . import utils
        return getattr(utils, name)
    if name in _PIPELINE:
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
