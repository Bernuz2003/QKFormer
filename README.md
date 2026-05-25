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


## Notice: Implementation Issue in SSA

### Issue Description & Fix

See the related issue: https://github.com/zhouchenlin2096/QKFormer/issues/14

We identified an implementation issue in the SSA module of the original repository. Specifically, the last line of the original implementation was:

```python
x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, W, H)
``` 
In this implementation, the LIF neuron is applied before reshaping the tensor back to the shape of (T, B, C, W, H), which may unintentionally mix the membrane potentials of different samples. The correct implementation should first restore the tensor shape and then apply the LIF neuron:

```python
x = self.proj_lif(self.proj_bn(self.proj_conv(x)).reshape(T, B, C, W, H))
``` 

### Impact on Reported Results

Our reproduced experiments indicate that this issue can lead to somewhat overestimated results on ImageNet. In contrast, its impact on the other four datasets—CIFAR-10, CIFAR-100, CIFAR10-DVS, and DVS128—is minimal, with only negligible differences observed between the two implementations.

This implementation issue does not affect the validity of the core methodological designs proposed in the paper, including the hybrid spiking attention mechanism, Spiking Patch Embedding with Deformed Shortcut (SPEDS), and the hierarchical spiking architecture. The method remains effective, and related architectural designs have been further supported and extended by subsequent studies [1,2,3]. Therefore, while this implementation issue affects part of the reported numerical results, particularly those on ImageNet, it does not alter the main conclusion of the paper.


### Recommendation for Future Research
For researchers building upon this work, **we strongly recommend using MaxFormer (https://github.com/bic-L/MaxFormer) [1], a variant of QKFormer, as the research subject. MaxFormer inherits the architectural design of QKFormer and uses an implementation that avoids the SSA issue described above.**


[1] Spiking Neural Networks Need High-Frequency Information, NeurIPS 2025.

[2] Scaling Spike-driven Transformer with Efficient Spike Firing Approximation Training.

[3] SpikingBrain: Spiking Brain-inspired Large Models.

