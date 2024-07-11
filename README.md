# Video-to-Audio Generation with Hidden Alignment  
Manjie Xu, Chenxing Li, Yong Ren, Rilin Chen, Yu Gu, Wei Liang, Dong Yu  
Tencent AI Lab  

<a href='https://arxiv.org/abs/2407.07464'>
  <img src='https://img.shields.io/badge/Paper-Arxiv-green?style=plastic&logo=arXiv&logoColor=green' alt='Paper Arxiv'>
</a>
<a href='https://sites.google.com/view/vta-ldm/home'>
  <img src='https://img.shields.io/badge/Project-Page-blue?style=plastic&logo=Google%20chrome&logoColor=blue' alt='Project Page'>
</a>  

Generating semantically and temporally aligned audio content in accordance with video input has become a focal point for researchers, particularly following the remarkable breakthrough in text-to-video generation. We aim to offer insights into the video-to-audio generation paradigm.

## Install
First install the python requirements. We recommend using conda:

```
conda create -n vta-ldm python=3.10
conda activate vta-ldm
pip install -r requirements.txt
```
Then download the checkpoints from [huggingface](https://huggingface.co/ariesssxu/vta-ldm-clip4clip-v-large), we recommend using git lfs:
```
mkdir ckpt && cd ckpt
git clone https://huggingface.co/ariesssxu/vta-ldm-clip4clip-v-large
# pull if large files are skipped:
cd vta-ldm-clip4clip-v-large && git lfs pull
```

## Model List
- ✅ VTA_LDM (the base model)  
- 🕳️ VTA_LDM+IB/LB/CAVP/VIVIT  
- 🕳️ VTA_LDM+text  
- 🕳️ VTA_LDM+PE
- 🕳️ VTA_LDM+text+concat  
- 🕳️ VTA_LDM+pretrain+text+concat  

## Inference
Put the video pieces into the `data` directory. Run the provided inference script to generate audio content from the input videos:
```
bash inference_from_video.sh
```
You can custom the hyperparameters to fit your personal requirements. We also provide a script that can help merge the generated audio content with the original video based on ffmpeg:

```
bash tools/merge_video_audio
```
## Training 
TBD. Code Coming Soon.

## Ack
This work is based on some of the great repos:  
[diffusers](https://github.com/huggingface/diffusers)  
[Tango](https://github.com/declare-lab/tango)  
[Audioldm](https://github.com/haoheliu/AudioLDM)  

## Cite us
```
@misc{xu2024vta-ldm,  
      title={Video-to-Audio Generation with Hidden Alignment},   
      author={Manjie Xu and Chenxing Li and Yong Ren and Rilin Chen and Yu Gu and Wei Liang and Dong Yu},
      year={2024},
      eprint={2407.07464},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2407.07464}, 
}
```
## Disclaimer

This is not an official product by Tencent Ltd.

