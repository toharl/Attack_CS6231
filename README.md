# Attack in collaborative settings.
Assuming we have N datasets that collaborate to train a classifier in one server. Some of the collaborators are able to send poisoned or redundant data (duplicates).
We consider the case of using loss-based filter against poison attack and collaborative settings and show that this model is vulnerable to a simple redundant attack.
Our simple attack is a dataset consists of duplicate samples.

This code is modified from the project: https://github.com/kuangliu/pytorch-cifar that trains CIFAR10 for classification. In our version we have added the attacks (ours and poison attack) and also implmented a filter algorithm against poison attack.


## First line of experiments:
N datasets (from N contributors), one of them consists of duplicates of one sample.
CIFAR10 consists of 50,000 training samples and 10,000 test samples.
We divide the training samples to N contributors. d is the size  of each dataset. d_a is the size of the attacker dataset, d_i is the size of dataset i.
In the first line of experiments the d=d_i=d_a=10,000, N=6 (5 trusted datasets and one attacker). f which is the fraction of small losses different for each run.
we generate the attacker dataset by randomly choose one sample and duplicate it (with the correct label).
Architecture -EfficientNetB0 (few CNN layers…)
Best in 150 epochs
filter fraction =1 (equivalent to no filter)

### To reproduce the results of the first line of experiments:
for example, our attack:
`python main.py --lr=0.01 --f=0.8 --attack=True`
for example, poison attack:
`python main.py --lr=0.01 --f=0.8 --poison=True`

Note that by default: N=6, num of epochs =150, size of attacker dataset is 50k/(N-1). you can also specify --d_a for using different size of attacker dataset.
# Original Readme:
# Train CIFAR10 with PyTorch

I'm playing with [PyTorch](http://pytorch.org/) on the CIFAR10 dataset.

## Prerequisites
- Python 3.6+
- PyTorch 1.0+

## Accuracy
| Model             | Acc.        |
| ----------------- | ----------- |
| [VGG16](https://arxiv.org/abs/1409.1556)              | 92.64%      |
| [ResNet18](https://arxiv.org/abs/1512.03385)          | 93.02%      |
| [ResNet50](https://arxiv.org/abs/1512.03385)          | 93.62%      |
| [ResNet101](https://arxiv.org/abs/1512.03385)         | 93.75%      |
| [MobileNetV2](https://arxiv.org/abs/1801.04381)       | 94.43%      |
| [ResNeXt29(32x4d)](https://arxiv.org/abs/1611.05431)  | 94.73%      |
| [ResNeXt29(2x64d)](https://arxiv.org/abs/1611.05431)  | 94.82%      |
| [DenseNet121](https://arxiv.org/abs/1608.06993)       | 95.04%      |
| [PreActResNet18](https://arxiv.org/abs/1603.05027)    | 95.11%      |
| [DPN92](https://arxiv.org/abs/1707.01629)             | 95.16%      |

## Learning rate adjustment
I manually change the `lr` during training:
- `0.1` for epoch `[0,150)`
- `0.01` for epoch `[150,250)`
- `0.001` for epoch `[250,350)`

Resume the training with `python main.py --resume --lr=0.01`
