# Assignment1
## Train a network on CIFAR10 dataset
Prepare envirenment(I am used to ues pytorch to implement.)
```
pip install -r requirements.txt
```
How to run(I train the model with a sigle 2080Ti.)
```
CUDA_VISIBLE_DEVICES=0 python train.py
```
Modify hyperparameters in config.yaml as your will

With hyperparameters as follows, I can get a accuracy rate of 80.7%
```
lr: 0.001
max_epoch: 50
weight_decay: 0.01
optimizer: SGD
momentum: 0.9
betas1: 0.9
betas2: 0.99
net: resnet18
batch_size: 32
```
I use resnet18(resnet50 and rest101 are also fine.) to classify the image. I train the net with a learning reate of 0.001 for 50 epochs. For the optimizer, I choice SGD, you can also use Adam or AdamW and set betas1 and betas2. In case of overfitting I set weight decay of 0.01. However, it also overfits.
