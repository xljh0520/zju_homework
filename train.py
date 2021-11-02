import torchvision
from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import munch
import yaml
from utils import AverageMetric
torch.backends.cudnn.benchmark = True

def main(args):

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
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            num_workers=8)

    test_loader = DataLoader(dataset=cifar10_test,
                            batch_size=args.batch_size,
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True,
                            num_workers=8)

    resnet = getattr(torchvision.models.resnet, args.net)
    net = resnet(pretrained=False)
    net.fc = nn.Linear(net.fc.in_features, 10)
    net.cuda()
    optimizer = getattr(torch.optim, args.optimizer)

    if 'adam' in args.optimizer.lower():
        optimizer = optimizer(net.parameters(), lr=float(args.lr),
                        weight_decay=float(args.weight_decay),
                        betas=(args.betas1, args.betas2))
    elif 'sgd' in  args.optimizer.lower():
        optimizer = optimizer(net.parameters(), lr=float(args.lr),
                        weight_decay=float(args.weight_decay), momentum=args.momentum)
    else:
        raise ValueError("ValueError optimizer should be [AdamW / Adam / SGD]")

    metric = nn.CrossEntropyLoss(reduction='mean')
    bset_acc = 0

    for epoch in range(1,args.max_epoch+1):
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
            train_loss.updata(bs, loss.item())
            if iter % 500 == 0:
                print('epoch: %d || iter: %d || loss: %.4f'%(epoch, iter, loss.item()))
        print('epoch: %d || train average loss: %.4f'%(epoch, train_loss.get_avg()))
        # eval
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
                test_loss.updata(bs, loss.item())
        print('epoch: %d || test average loss: %.4f'%(epoch, test_loss.get_avg()))
        cur_acc = acc/num_imgs
        print('cur acc: %.4f'%(cur_acc))
        bset_acc = cur_acc if cur_acc > bset_acc else bset_acc
        print('bset acc:%.4f'%(bset_acc))
        print('='*50)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Config file')
    parser.add_argument('-c', '--config', help='path to config file', required=True)
    arg = parser.parse_args()
    config_path = arg.config
    args = munch.munchify(yaml.safe_load(open(config_path)))
    print(args)
    main(args)

     
