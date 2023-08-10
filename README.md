# SAILOR: Structural Augmentation Based Tail Node Representation Learning
We provide the code (in pytorch) for our paper "SAILOR: Structural Augmentation Based Tail Node Representation Learning" (SAILOR for short), which is published in CIKM-2023.

## 1. Installation Guide
The following commands are used for installing key dependencies; other can be directly installed via pip or conda. A full redundant dependency list is in `requirements.txt`.
```
pip install dgl
pip install torch==1.12.0+cu102
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0%2Bcu102/torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.12.0%2Bcu102/torch_sparse-0.6.15%2Bpt112cu102-cp39-cp39-linux_x86_64.whl
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.12.0%2Bcu102/torch_cluster-1.6.0%2Bpt112cu102-cp39-cp39-linux_x86_64.whl
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.12.0%2Bcu102/torch_spline_conv-1.2.1%2Bpt112cu102-cp39-cp39-linux_x86_64.whl
pip install torch-geometric
```

## 2. Runing Examples 
```
python main.py --config configs/config_citeseer_sailor_arch1.yaml --outpth output/citeseer_arch1 --seed 0
python main.py --config configs/config_citeseer_sailor_arch2.yaml --outpth output/citeseer_arch2 --seed 0
python main.py --config configs/config_citeseer_sailor_arch3.yaml --outpth output/citeseer_arch3 --seed 0
python main.py --config configs/config_citeseer_sailor_arch1_public.yaml --outpth output/citeseer_arch1_public --seed 0
```

## 3. Cite
```
@inproceedings{
author = {Liao, Jie and Li, Jintang and Chen, Liang and Wu, Bingzhe and Bian, Yatao and Zheng, Zibin},
title = {SAILOR: Structural Augmentation Based Tail Node Representation Learning},
year = {2023},
publisher = {ACM},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3583780.3615045},
doi = {10.1145/3583780.3615045},
booktitle = {Proceedings of the 32nd ACM International Conference on Information & Knowledge Management},
series = {CIKM '23}
}
```
