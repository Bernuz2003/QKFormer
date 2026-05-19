# QKFormer: Hierarchical Spiking Transformer using Q-K Attention 


## Abstact

Spiking Transformers, which integrate Spiking Neural Networks (SNNs) with Transformer architectures, have attracted significant attention due to their potential for low energy consumption and high performance. 
However, there remains a substantial gap in performance between SNNs and Artificial Neural Networks (ANNs). To narrow this gap, we have developed QKFormer, a direct training spiking transformer with the following features: 
i) Linear complexity and high energy efficiency, the novel spike-form Q-K attention module efficiently models the token or channel attention through binary vectors and enables the construction of larger models.
ii) Multi-scale spiking representation, achieved by a hierarchical structure with the different number of tokens across blocks. 
iii) Spiking Patch Embedding with Deformed Shortcut (SPEDS), enhances spiking information transmission and integration, thus improving overall performance. 
%Together, we develop QKFormer, a hierarchical spiking transformer based on Q-K attention with direct training. 
It is shown that QKFormer achieves significantly superior performance over existing state-of-the-art SNN models on various mainstream datasets. 

<p align="center">
<img src="https://github.com/zhouchenlin2096/QKFormer/blob/master/imgs/QKFormer.png">
</p>


## Requirements

```
timm==0.6.12
cupy==11.4.0
torch==1.12.1
spikingjelly==0.0.0.0.12
pyyaml
tensorboard
```

data prepare: ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Train & Test
### Training  on ImageNet
```
cd imagenet
python -m torch.distributed.launch --nproc_per_node=8 train.py
```

### Testing ImageNet Val data
Download the trained model first, then:
```
cd imagenet
python test.py
```

### Training  on CIFAR10
Setting hyper-parameters in cifar10.yml
```
cd cifar10
python train.py
```

### Training  on CIFAR100
Setting hyper-parameters in cifar100.yml
```
cd cifar10
python train.py
```

### Training  on DVS128 Gesture
```
cd dvs128-gesture
python train.py
```

### Training  on CIFAR10-DVS
```
cd cifar10-dvs
python train.py
```

## Reference
If you find this repo useful, please consider citing:
```
@inproceedings{
zhou2024qkformer,
title={{QKF}ormer: Hierarchical Spiking Transformer using Q-K Attention},
author={Chenlin Zhou and Han Zhang and Zhaokun Zhou and Liutao Yu and Liwei Huang and Xiaopeng Fan and Li Yuan and Zhengyu Ma and Huihui Zhou and Yonghong Tian},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=AVd7DpiooC}
}

@article{zhou2024direct,
  title={Direct training high-performance deep spiking neural networks: a review of theories and methods},
  author={Zhou, Chenlin and Zhang, Han and Yu, Liutao and Ye, Yumin and Zhou, Zhaokun and Huang, Liwei and Ma, Zhengyu and Fan, Xiaopeng and Zhou, Huihui and Tian, Yonghong},
  journal={Frontiers in Neuroscience},
  volume={18},
  pages={1383844},
  year={2024},
  publisher={Frontiers Media SA}
}

@article{zhang2024sglformer,
  title={SGLFormer: Spiking Global-Local-Fusion Transformer with high performance},
  author={Zhang, Han and Zhou, Chenlin and Yu, Liutao and Huang, Liwei and Ma, Zhengyu and Fan, Xiaopeng and Zhou, Huihui and Tian, Yonghong},
  journal={Frontiers in Neuroscience},
  volume={18},
  pages={1371290},
  year={2024},
  publisher={Frontiers Media SA}
}

@article{zhou2023spikingformer,
  title={Spikingformer: Spike-driven residual learning for transformer-based spiking neural network},
  author={Zhou, Chenlin and Yu, Liutao and Zhou, Zhaokun and Ma, Zhengyu and Zhang, Han and Zhou, Huihui and Tian, Yonghong},
  journal={arXiv preprint arXiv:2304.11954},
  year={2023}
}
```


## Acknowledgement

We recommend using [MaxFormer](https://github.com/bic-L/MaxFormer), a variant of QKFormer, as the research subject — a model that inherits QKFormer's architecture yet successfully addresses its shortcomings.

