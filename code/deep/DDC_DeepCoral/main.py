import argparse
import torch
import os
import data_loader
import models
import utils
import numpy as np
from return_dataset import return_SSDA
import datetime
gpu_id = 1
DEVICE = torch.device('cuda:%d'%gpu_id if torch.cuda.is_available() else 'cpu')

log = []

# Command setting
parser = argparse.ArgumentParser(description='DDC_DCORAL')
parser.add_argument('--model', type=str, default='resnet50')
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--src', type=str, default='webcam')
parser.add_argument('--tar', type=str, default='amazon')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--n_epoch', type=int, default=100) #100
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', type=float, default=5e-4)
parser.add_argument('--data', type=str, default='/home/begum/SSDA_MME/data/')
parser.add_argument('--dataset', type=str, default='office',
                    choices=['multi', 'office', 'office_home'],
                    help='the name of dataset')
# parser.add_argument('--data', type=str, default='/home/begum/SSDA_MME/data/office/')
parser.add_argument('--early_stop', type=int, default=20)
parser.add_argument('--lamb', type=float, default=10)
parser.add_argument('--trans_loss', type=str, default='coral')
parser.add_argument('--checkpath', type=str, default='./logs/models')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
args = parser.parse_args()

if not os.path.exists(args.checkpath):
    os.makedirs(args.checkpath)

def save_mdl(model,pth, epoch):
    torch.save(model.state_dict(),os.path.join(args.checkpath,
                                        "mdl_{}_{}_"
                                        "to_{}_epoch_{}.pth".
                                        format(args.trans_loss, args.src,
                                               args.tar, epoch)))
    
def test(model, target_test_loader):
    model.eval()
    test_loss = utils.AverageMeter()
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        for data, target in target_test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            s_output = model.predict(data)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)
    acc = 100. * correct / len_target_dataset
    return acc


def train(source_loader, target_loader, target_train_loader, target_test_loader, model, optimizer):
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    len_target = len(target_loader)
   
    best_acc = 0
    stop = 0
    suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    for e in range(args.n_epoch):
        stop += 1
        train_loss_clf = utils.AverageMeter()
        train_loss_transfer = utils.AverageMeter()
        train_loss_total = utils.AverageMeter()
        model.train()
        iter_source, iter_target_train, iter_target = iter(source_loader), iter(target_train_loader), iter(target_loader)
        n_batch = min(len_source_loader, len_target_loader)
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(n_batch):
            data_source, label_source = iter_source.next()
            data_target_train, label_target = iter_target_train.next()
            data_target, _ = iter_target.next()
            
            data_source, label_source = data_source.to(
                DEVICE), label_source.to(DEVICE)
            data_target_train, label_target = data_target_train.to(
                DEVICE), label_target.to(DEVICE)
            data_target = data_target.to(DEVICE)
            optimizer.zero_grad()
            label_pred, transfer_loss = model(data_source, data_target, data_target_train)
            clf_loss = criterion(label_pred, torch.cat(((label_source,label_target)),dim=0))
            loss = clf_loss + args.lamb * transfer_loss
            loss.backward()
            optimizer.step()
            train_loss_clf.update(clf_loss.item())
            train_loss_transfer.update(transfer_loss.item())
            train_loss_total.update(loss.item())
        # Test
        acc = test(model, target_test_loader)
        log.append([train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg, acc])
        np_log = np.array(log, dtype=float)
        
        np.savetxt('logs/train_log'+suffix+'.csv', np_log, delimiter=',', fmt='%.6f',
                header='Dataset %s Source %s Target %s Labeled num perclass %s Network %s Lamb %d trans_loss %s' %
                (args.dataset, args.src, args.tar, args.num, args.model, args.lamb, args.trans_loss))
        print('Epoch: [{:2d}/{}], cls_loss: {:.4f}, transfer_loss: {:.4f}, total_Loss: {:.4f}, acc: {:.4f}'.format(
                    e, args.n_epoch, train_loss_clf.avg, train_loss_transfer.avg, train_loss_total.avg, acc))
        if best_acc < acc:
            best_acc = acc
            stop = 0
        if stop >= args.early_stop:
            break
    print('Transfer result: {:.4f}'.format(best_acc))
    

def load_data(src, tar, root_dir):
    folder_src = os.path.join(root_dir, src)+'/images/'
    folder_tar = os.path.join(root_dir, tar)+'/images/'
    source_loader = data_loader.load_data(
        folder_src, args.batchsize, True, {'num_workers': 4})
    target_train_loader = data_loader.load_data(
        folder_tar, args.batchsize, True, {'num_workers': 4})
    target_test_loader = data_loader.load_data(
        folder_tar, args.batchsize, False, {'num_workers': 4})
    return source_loader, target_train_loader, target_test_loader


if __name__ == '__main__':
    torch.manual_seed(0)

    print('Src: %s, Tar: %s' % (args.src, args.tar))

    source_loader, target_loader, target_test_loader, target_train_loader, class_list = return_SSDA(args)

    model = models.Transfer_Net(
        len(class_list), transfer_loss=args.trans_loss, base_net=args.model, gpu_id = gpu_id).to(DEVICE)
    optimizer = torch.optim.SGD([
        {'params': model.base_network.parameters()},
        {'params': model.bottleneck_layer.parameters(), 'lr': 10 * args.lr},
        {'params': model.classifier_layer.parameters(), 'lr': 10 * args.lr},
    ], lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    train(source_loader, target_loader, target_train_loader,
          target_test_loader, model, optimizer)
