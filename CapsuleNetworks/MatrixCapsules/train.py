from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from model import capsules
from loss import SpreadLoss


def load_database(args):
    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    path = os.path.join(args.data_folder, args.dataset)
    if args.dataset == 'mnist':
        num_class = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(path, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.dataset == 'cifar10':
        num_class = 10
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(path, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor()
                               ])), 
                            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(path, train=False, transform=transforms.Compose([
                                transforms.ToTensor()
                               ])),
                            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    else:
        raise NameError('Undefined dataset {}'.format(args.dataset))
    return num_class, train_loader, test_loader


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    train_len = len(train_loader)
    epoch_acc = 0
    epoch_loss = 0
    count = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        _s = time.time()

        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        r = min(10.0, batch_idx / train_len - 4)
        target = target.to(device)
        loss = criterion(output, target, r)
        acc = accuracy(output, target)
        loss.backward()
        optimizer.step()
        
        _e = time.time()
        
        epoch_acc += acc[0].item()
        epoch_loss += loss.item()
        assert epoch_loss > 0
        count += 1
        if batch_idx % args.log_interval == 0:
            print('Train epoch: {}\t[{}/{} ({:.0f}%)]\t'
                'Loss: {:.6f}\tAccuracy: {:.6f}\t'
                'Last batch time {batch_time:.3f}\t'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), 100.0 * batch_idx / len(train_loader),
                epoch_loss / count, epoch_acc / count,
                batch_time = _e - _s))
    return epoch_acc, epoch_loss


def save(model, epoch, optimizer, criterion, scheduler, accs, accs_test, loss, loss_test, folder, A, B, C, D):
    path = os.path.join(folder, optimizer.__class__.__name__ + "_" + str(A) + str(B) + str(C) + str(D) + "_iters" + str(args.em_iters) + "_lr" + str(args.lr) + "_batch" + str(args.batch_size) + "_model.pth")
     if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    print("Saving model to " + str(path) + " on epoch: " + str(epoch))
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'criterion': criterion.state_dict(),
        'scheduler': scheduler.state_dict(),
        'accs': accs,
        'accs_test': accs_test,
        'loss': loss,
        'loss_test': loss_test
    }
    torch.save(state, path)
    print("Saved.")

def load(model, epoch, optimizer, criterion, scheduler, accs, accs_test, loss, loss_test, folder, A, B, C, D):
    path = os.path.join(folder, optimizer.__class__.__name__ + "_" + str(A) + str(B) + str(C) + str(D) + "_iters" + str(args.em_iters) + "_lr" + str(args.lr) + "_batch" + str(args.batch_size) + "_model.pth")
    if not os.path.exists(path):
        return model, epoch, optimizer, criterion, scheduler, accs, accs_test, loss, loss_test
    print("Loading the model...")
    state = torch.load(path)
    epoch = state['epoch']
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    criterion.load_state_dict(state['criterion'])
    scheduler.load_state_dict(state['scheduler'])
    accs = state['accs']
    accs_test = state['accs_test']
    loss = state['loss']
    loss_test = state['loss_test']
    print("Loaded. Resuming on epoch: " + str(epoch))
    print(str(model))
    print(str(optimizer))
    return model, epoch, optimizer, criterion, scheduler, accs, accs_test, loss, loss_test

def test(test_loader, model, criterion, device):
    print("Evaluating...")
    _s = time.time()
    model.eval()
    test_loss = 0
    acc = 0
    test_len = len(test_loader)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target, r=1).item()
            acc += accuracy(output, target)[0].item()

    test_loss /= test_len
    acc /= test_len
    assert test_loss > 0
    print("Test: Average loss: {:.6f}, Accuracy: {:.6f}, \t Time:{:.3f}".format(test_loss, acc, round(time.time() - _s, 3)))
    return acc, test_loss

def plot_graph(line1, line2, type):
    plt.plot(line1, label="Train " + type)
    plt.plot(line2, label="Test " + type)
    plt.ylabel(type)
    plt.xlabel("Epoch")
    plt.legend()
    xtick = np.arange(0, len(line1), 1)
    plt.xticks(xtick)
    plt.tight_layout()
    plt.show()

class SETTINGS:  
  def __init__(self):
    self.batch_size = 24
    self.test_batch_size = 24
    self.em_iters = 3
    self.lr = 0.0003
    self.weight_decay = 0.000000002
    self.epochs = 300
    self.cuda = True
    self.seed = 1
    self.log_interval = 100
    self.save_folder = 'D:\GDrive\MASTERS\Semestre 2\CAPS\saves'
    self.data_folder = './data'
    self.dataset = 'cifar10'

def main():

    A, B, C, D = 256, 22, 22, 22
   

    global args
    args = SETTINGS()
    args.cuda = args.cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")
    
    num_class, train_loader, test_loader = load_database(args)

    model = capsules(A=A, B=B, C=C, D=D, E=num_class, iters=args.em_iters).to(device)

    criterion = SpreadLoss(num_class=num_class, m_min=0.2, m_max=0.9)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=args.weight_decay, nesterov=True)
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=1, verbose=True)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma= 0.81, last_epoch=-1)

    epoch = 1
    _accs = []
    _accs_test = []
    _loss = []
    _loss_test = []
    model, epoch, optimizer, criterion, scheduler, _accs, _accs_test, _loss, _loss_test = load(model, epoch, optimizer, criterion, scheduler, _accs, _accs_test, _loss, _loss_test, args.save_folder, A, B, C, D)

    #optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)

    #plot_graph(_accs, _accs_test, "Accuracy")
    #plot_graph(_loss, _loss_test, "Loss")
    #quit()

    #best_acc = test(test_loader, model, criterion, device)
    #quit()
    print(str(model))
    print(str(optimizer))

    try:
        for epoch in range(epoch, args.epochs + 1):
            _s = time.time()
            acc, loss = train(train_loader, model, criterion, optimizer, epoch, device)
            acc /= len(train_loader)
            _accs.append(acc)
            loss /= len(train_loader)
            _loss.append(loss)
            print("Average acc: " + str(acc))
            print("Average loss: " + str(loss))
            print("Epoch time: " + str(round(time.time() - _s, 3)))
            test_acc, test_loss = test(test_loader, model, criterion, device)
            _accs_test.append(test_acc)
            _loss_test.append(test_loss)
            save(model, epoch+1, optimizer, criterion, scheduler, _accs, _accs_test, _loss, _loss_test, args.save_folder, A, B, C, D)
            #scheduler.step(acc)
            #print(optimizer)
        plot_graph(_accs, _accs_test, "Accuracy")
        plot_graph(_loss, _loss_test, "Loss")
    except KeyboardInterrupt:
        if len(_accs_test) < len(_accs):
            test_acc, test_loss = test(test_loader, model, criterion, device)
            _accs_test.append(test_acc)
            _loss_test.append(test_loss)
        #    epoch += 1
        #save(model, epoch, optimizer, criterion, scheduler, _accs, _accs_test, _loss, _loss_test, args.save_folder, A, B, C, D)
        plot_graph(_accs, _accs_test, "Accuracy")
        plot_graph(_loss, _loss_test, "Loss")
        quit()

if __name__ == '__main__':
    main()
