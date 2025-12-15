# XIDLE-Net
This is the implementation of the paper ["Inspector gaze-guided multitask learning for explainable structural damage assessment"](https://onlinelibrary.wiley.com/doi/10.1111/mice.70131).

## Getting Started
### Installation
* Clone this repo:
~~~~
git clone https://github.com/itschenyu/XIDLE-Net.git
cd XIDLE-Net
~~~~
### Dataset
* Please download the dataset from [here](https://drive.google.com/drive/folders/1D0XnvuNAHacTRnIhOvPdk7PCpPJnc_v7?usp=sharing) and then place it in `./XIDLE-Net/dataset`.

### Pre-trained Weight
* Please download pre-trained weights on ImageNet-22K from [here](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth) and place it in `./XIDLE-Net/model_data/`.

### Model Download
* Please download the XIDLE-Net model weight from [here](https://drive.google.com/file/d/1AFHo5VRv_bwVzMNUIf0nOWFElAzpAUCQ/view?usp=sharing).

### Training
~~~~
python RUN.py
~~~~

### Testing
Evaluating the model on the test set:
~~~~
python TEST.py
~~~~

## Citation
If XIDLE-Net and the eye gaze dataset are helpful to you, please cite them as:
~~~~
@article{https://doi.org/10.1111/mice.70131,
  author = {Zhang, Chenyu and Liu, Charlotte and Li, Ke and Yin, Zhaozheng and Qin, Ruwen},
  title = {Inspector gaze-guided multitask learning for explainable structural damage assessment},
  journal = {Computer-Aided Civil and Infrastructure Engineering},
  volume = {40},
  number = {30},
  pages = {5824-5841},
  doi = {https://doi.org/10.1111/mice.70131},
  url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/mice.70131},
  eprint = {https://onlinelibrary.wiley.com/doi/pdf/10.1111/mice.70131},
  year = {2025}
}
~~~~
## Note
Part of the codes are referred from <a href="https://github.com/hz-zhu/MT-UNet/">MT-UNet</a> project.

The images and damage level labels in the dataset are credited to [PEER Hub ImageNet](https://apps.peer.berkeley.edu/phi-net/).
