import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from torchvision.models.resnet import resnet50 as resnet
torch.backends.cudnn.benchmark = True

class AverageMetric(object):
    def __init__(self):
        self.cnt_list = []
        self.val_list = []
    def get_avg(self):
        sum_cnt = 0
        sum_val = 0
        for val, cnt in zip(self.val_list, self.cnt_list):
            sum_val += val*cnt
            sum_cnt += cnt
        return sum_val / sum_cnt
    def reset(self):
        self.cnt_list = []
        self.val_list = []
    def updata(self, cnt, val):
        self.cnt_list.append(cnt)
        self.val_list.append(val)

cifar10 = torchvision.datasets.CIFAR10(
    root='./datasets',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(size=(32, 32)),
        transforms.ToTensor(),
    ])
)
cifar10_test = torchvision.datasets.CIFAR10(
    root='./datasets',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
    ])
)

train_loader = DataLoader(dataset=cifar10,
                          batch_size=32,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True,
                          num_workers=8)
                                              
test_loader = DataLoader(dataset=cifar10_test,
                          batch_size=32,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True,
                          num_workers=8)

net = resnet(pretrained=False)
net.fc = nn.Linear(2048, 10)

net.cuda()

max_epoch = 30
optimizer = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=5e-4)
metric = nn.CrossEntropyLoss(reduction='mean')
bset_acc = 0

for epoch in range(1, max_epoch+1):
    # train
    net.train()
    train_loss = AverageMetric()
    for iter, data in enumerate(train_loader):
        optimizer.zero_grad()
        img, lab = data
        img = img.cuda()
        lab = lab.long().cuda()
        bs = img.size(0)
        pred = net(img)
        loss = metric(pred, lab)
        loss.backward()
        optimizer.step()
        train_loss.updata(bs, loss)
        if iter % 500 == 0:
            print('epoch: %d || iter: %d || loss: %.4f'%(epoch, iter, loss.item()))
    
    print('epoch: %d || train average loss: %.4f'%(epoch, train_loss.get_avg()))
    #eval
    net.eval()
    num_imgs = len(cifar10_test)
    acc = 0
    test_loss = AverageMetric()
    
    with torch.no_grad():
        for iter, data in enumerate(test_loader):
            img, lab = data
            img = img.cuda()
            lab = lab.long().cuda()
            pred = net(img)
            loss = metric(pred, lab)
            pred = torch.argmax(pred,dim=-1)
            acc += pred.eq(lab).sum().item()
            test_loss.updata(bs, loss)
    
    print('epoch: %d || test average loss: %.4f'%(epoch, test_loss.get_avg()))

    cur_acc = acc/num_imgs
    print('cur acc: %.4f'%(cur_acc))
    bset_acc = cur_acc if cur_acc > bset_acc else bset_acc
    print('bset acc:%.4f'%(bset_acc))
    print('='*100)