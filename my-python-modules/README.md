# White Mold Detection Transformer (DETR) Model

## Institute of Computing (IC) at University of Campinas (Unicamp)

## Postgraduate Program in Computer Science

### Team

* Rubens de Castro Pereira - student at IC-Unicamp
* Prof. Dr. Hélio Pedrini - advisor at IC-Unicamp
* Prof. Dr. Díbio Leandro Borges - coadvisor at CIC-UnB
* Prof. Dr. Murillo Lobo Jr. - coadvisor at Embrapa Rice and Beans

### Main purpose

This Python project aims to train and inference the DETR model in the image dataset of white mold disease and its stages.

This implementation is based on this references: 
* Title: L-7 | DETR | Object detection Using Detection Transformer on custom dataset
* Author: Aarohi Singla
* YouTube: https://www.youtube.com/watch?v=90tWnm9VfLI
* Notebook: https://github.com/AarohiSingla/Detection-Transformer/blob/main/detr_implementation.ipynb
* Github: https://github.com/AarohiSingla/Detection-Transformer

## Installing Python Virtual Environment

```
module load python/3.10.10-gcc-9.4.0
```
```
pip install --user virtualenv
```
```
virtualenv -p python3.10 venv-wm-model-detr
```
```
source venv-wm-model-detr/bin/activate
```
```
pip install -r requirements.txt
```

## Preparing models to offline mode

```
install 'huggingface-cli' according by "https://huggingface.co/docs/huggingface_hub/en/guides/cli"
```
```
use huggingface client to download models to local cache
- huggingface-cli download facebook/detr-resnet-50
- huggingface-cli download timm/resnet50.a1_in1k
```

## Running Python Application

```
access specific folder 'wm-model-detr'
```
```
python my-python-modules/manage_detr_train.py
```

## Submitting Python Application in the LoveLace environment at CENAPAD (firewalled)

Version of CUDA module to load:
- module load cuda/11.5.0-intel-2022.0.1

```
qsub wm-model-detr.script
```
```
qstat -u rubenscp
```
```
qstat -q umagpu
```

The results of job execution can be visualizedat some files as:

* errors
* output

## Troubleshootings

- In the first execution, some files are downloaded to the CENAPAD environment, and it's possible that thgis operation can't be done automatically because the security rules. So, you must identify the files need to download in the right place and do it manually by 'wget' command. For example: 
    - cd ~/.config/Ultralytics
    - wget https://ultralytics.com/assets/Arial.ttf
    