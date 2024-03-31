# DGHIN
The code of the paper "Dual-Graph Hierarchical Interaction Network for Referring Image Segmentation".  
 
## Requirements
We have tested the code on the following environment:  
```
Python==3.6.13  
Pytorch==1.7.1  
CUDA==11.0  
torch-geometric==2.0.3  
SpaCy==2.3.7  
en-core-web-lg==2.3.0  
```
## Data Preprocessing
We train our model on for datasets: RefCOCO/RefCOCO+/G-Ref/ReferIt, we have pre-processed the data for all datasets for data loading. Taking the data from the refcoco dataset as an example, run
```
python ./data_preprocess/data_preprocess_v3.py --data_root . --output_dir data_v3 --dataset refcoco --split unc --generate_mask 
```
If you want to re-implement the results of the jointly training version, please modify the data_preprocess_v3.py file.
Please download the offical pre-trained weights of the visual backbone (Swin-Base 22K 384x384):
```
https://github.com/microsoft/Swin-Transformer
```  
and the linguistic backbone (BERT-base-uncased): 
```
(model)https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin 
(json file)https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json 
(vocab_file) https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt  
```
The final data structure should be placed in the following structure：
```
.
├── bert-base-uncased
│   ├── config.json
│   ├── gitattributes
│   ├── pytorch_model.bin
│   ├── README.md
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
├── data_process_v3.py
├── data_v3
│   ├── anns
│   │   ├── refcoco
│   │   │   ├── testA.json
│   │   │   ├── testB.json
│   │   │   ├── train.json
│   │   │   └── val.json
│   │   ├── refcoco+
│   │   │   ├── testA.json
│   │   │   ├── testB.json
│   │   │   ├── train.json
│   │   │   └── val.json
│   │   ├── refcocog_google
│   │   │   ├── train.json
│   │   │   └── val.json
│   │   ├── refcocog_umd
│   │   │   ├── test.json
│   │   │   ├── train.json
│   │   │   └── val.json
│   │   └── referit
│   │       ├── test.json
│   │       └── trainval.json
│   └── masks
│       ├── refcoco
│       │   ├── 0_0.png
│       │   ├──    ....
│       │   └── 9999_2.png
│       ├── refcoco+
│       │   ├── 0_0.png
│       │   ├──    ....
│       │   └── 9999_2.png
│       ├── refcocog_google
│       │   ├── 0_0.png
│       │   ├──    ....
│       │   └── 9999_1.png
│       ├── refcocog_umd
│       │   ├── 0_0.png
│       │   ├──    ....
│       │   └── 9999_1.png
│       └── referit
│           ├── 0_0.png
│           ├──     ....
│           └── 9999_0.png
├── images
│   ├── referit
│   │   ├── images
│   │   │   ├── 10000.jpg
│   │   │   ├──    ....
│   │   │   └── 999.jpg
│   │   └── mask
│   │       ├── 10000_1.mat
│   │       ├──      ....
│   │       └── 9999_2.mat
│   └── train2014
│       ├── COCO_train2014_000000000009.jpg
│       ├──                      ....
│       └── COCO_train2014_000000581921.jpg
├── pretrained_weights
│   ├── swin_base_patch4_window12_384_22k.pth
├── __pycache__
│   └── refer.cpython-36.pyc
├── refcoco
│   ├── instances.json
│   ├── refs(google).p
│   └── refs(unc).p
├── refcoco+
│   ├── instances.json
│   └── refs(unc).p
├── refcocog
│   ├── instances.json
│   ├── refs(google).p
│   └── refs(umd).p
├── referit
│   ├── referit_all_imlist.txt
│   ├── referit_bbox.json
│   ├── referit_imcrop.json
│   ├── referit_imsize.json
│   ├── referit_query_all.json
│   ├── referit_query_test.json
│   ├── referit_query_train.json
│   ├── referit_query_trainval.json
│   ├── referit_query_val.json
│   ├── referit_test_imlist.txt
│   ├── referit_train_imlist.txt
│   ├── referit_trainval_imlist.txt
│   └── referit_val_imlist.txt
├── refer.py
```
## Usage
Please change the path in the config files before running the code.  
### Train  
We use `DistributedDataParallel` from PyTorch.
The released model was trained using 4 x 24G RTX3090 cards  
Train the model using the settings in the paper, run：
```
CUDA_VISIBLE_DEVICES = 0,1,2,3 python -m torch.distributed.launch --nproc_per_node = 4 train.py
```
### Test
Test the model with the trained weights, run:
```
python eval.py --trained_model results/xx.pth
```
