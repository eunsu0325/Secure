import os
import argparse
import time
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from torch.optim import lr_scheduler
import cv2 as cv
import numpy as np
from loss import SupConLoss
from models import MyDataset
from models.ccnet import ccnet
from utils import *
from config import ConfigParser

import copy


def test(model, config):
    print('Start Testing!')
    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))

    path_hard = os.path.join(config.training.results_path, 'rank1_hard')

    trainset = MyDataset(txt=config.dataset.train_set_file, transforms=None, train=False)
    testset = MyDataset(txt=config.dataset.test_set_file, transforms=None, train=False)

    batch_size = 512  # 128

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=2)
    data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=2)

    fileDB_train = getFileNames(config.dataset.train_set_file)
    fileDB_test = getFileNames(config.dataset.test_set_file)

    # output dir
    if not os.path.exists(config.training.results_path):
        os.makedirs(config.training.results_path)

    if not os.path.exists(path_hard):
        os.makedirs(path_hard)

    net = model
    net.cuda()
    net.eval()

    # feature extraction:
    featDB_train = []
    iddb_train = []

    for batch_id, (datas, target) in enumerate(data_loader_train):
        data = datas[0]
        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode(data)
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_train = codes
            iddb_train = y
        else:
            featDB_train = np.concatenate((featDB_train, codes), axis=0)
            iddb_train = np.concatenate((iddb_train, y))

    print('completed feature extraction for training set.')
    print('featDB_train.shape: ', featDB_train.shape)

    classNumel = len(set(iddb_train))
    num_training_samples = featDB_train.shape[0]

    trainNum = num_training_samples // classNumel
    print('[classNumel, imgs/class]: ', classNumel, trainNum)
    print('\n')

    featDB_test = []
    iddb_test = []

    print('Start Test Feature Extraction.')
    for batch_id, (datas, target) in enumerate(data_loader_test):
        data = datas[0]
        data = data.cuda()
        target = target.cuda()

        codes = net.getFeatureCode(data)
        codes = codes.cpu().detach().numpy()
        y = target.cpu().detach().numpy()

        if batch_id == 0:
            featDB_test = codes
            iddb_test = y
        else:
            featDB_test = np.concatenate((featDB_test, codes), axis=0)
            iddb_test = np.concatenate((iddb_test, y))

    print('completed feature extraction.')
    print('featDB_test.shape: ', featDB_test.shape)
    print('\nFeature Extraction Done!')
    print('start feature matching ...\n')

    # verification EER of the test set
    s = []
    l = []
    ntest = featDB_test.shape[0]
    ntrain = featDB_train.shape[0]

    for i in range(ntest):
        feat1 = featDB_test[i]
        for j in range(ntrain):
            feat2 = featDB_train[j]
            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi
            s.append(dis)
            if iddb_test[i] == iddb_train[j]:
                l.append(1)
            else:
                l.append(-1)

    veriEER_path = str(config.training.results_path) + 'veriEER'
    if not os.path.exists(veriEER_path):
        os.makedirs(veriEER_path)
    if not os.path.exists(veriEER_path + '/rank1_hard/'):
        os.makedirs(veriEER_path + '/rank1_hard/')

    with open(veriEER_path + '/scores_VeriEER.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + veriEER_path + '/scores_VeriEER.txt scores_VeriEER')
    os.system('python ./getEER.py' + '  ' + veriEER_path + '/scores_VeriEER.txt scores_VeriEER')

    print('\n------------------')
    print('Rank-1 acc of the test set...')
    
    # rank-1 acc
    cnt = 0
    corr = 0
    for i in range(ntest):
        probeID = iddb_test[i]
        dis = np.zeros((ntrain, 1))
        for j in range(ntrain):
            dis[j] = s[cnt]
            cnt += 1
        idx = np.argmin(dis[:])
        galleryID = iddb_train[idx]

        if probeID == galleryID:
            corr += 1
        else:
            testname = fileDB_test[i]
            trainname = fileDB_train[idx]
            im_test = cv.imread(testname)
            im_train = cv.imread(trainname)
            img = np.concatenate((im_test, im_train), axis=1)
            cv.imwrite(veriEER_path + '/rank1_hard/%6.4f_%s_%s.png' % (
                np.min(dis[:]), testname[-13:-4], trainname[-13:-4]), img)

    rankacc = corr / ntest * 100
    print('rank-1 acc: %.3f%%' % rankacc)
    print('-----------')

    with open(veriEER_path + '/rank1.txt', 'w') as f:
        f.write('rank-1 acc: %.3f%%' % rankacc)

    # Real EER
    print('\n\nReal EER of the test set...')
    s = []
    l = []
    n = featDB_test.shape[0]
    for i in range(n - 1):
        feat1 = featDB_test[i]
        for jj in range(n - i - 1):
            j = i + jj + 1
            feat2 = featDB_test[j]
            cosdis = np.dot(feat1, feat2)
            dis = np.arccos(np.clip(cosdis, -1, 1)) / np.pi
            s.append(dis)
            if iddb_test[i] == iddb_test[j]:
                l.append(1)
            else:
                l.append(-1)

    print('feature extraction about real EER done!\n')

    with open(veriEER_path + '/scores_EER_test.txt', 'w') as f:
        for i in range(len(s)):
            score = str(s[i])
            label = str(l[i])
            f.write(score + ' ' + label + '\n')

    sys.stdout.flush()
    os.system('python ./getGI.py' + '  ' + veriEER_path + '/scores_EER_test.txt scores_EER_test')
    os.system('python ./getEER.py' + '  ' + veriEER_path + '/scores_EER_test.txt scores_EER_test')


def fit(epoch, model, data_loader, phase, optimizer, criterion, con_criterion, config):
    if phase not in ['training', 'testing']:
        raise TypeError('input error!')

    if phase == 'training':
        model.train()
    else:
        model.eval()

    running_loss = 0
    running_correct = 0

    for batch_id, (datas, target) in enumerate(data_loader):
        data = datas[0].cuda()
        data_con = datas[1].cuda()
        target = target.cuda()

        if phase == 'training':
            optimizer.zero_grad()
            output, fe1 = model(data, target)
            output2, fe2 = model(data_con, target)
            fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)
        else:
            with torch.no_grad():
                output, fe1 = model(data, None)
                output2, fe2 = model(data_con, None)
                fe = torch.cat([fe1.unsqueeze(1), fe2.unsqueeze(1)], dim=1)

        ce = criterion(output, target)
        ce2 = con_criterion(fe, target)
        loss = config.training.ce_weight * ce + config.training.contrastive_weight * ce2

        running_loss += loss.data.cpu().numpy()
        preds = output.data.max(dim=1, keepdim=True)[1]
        running_correct += preds.eq(target.data.view_as(preds)).cpu().sum().numpy()

        if phase == 'training':
            loss.backward(retain_graph=None)
            optimizer.step()

    total = len(data_loader.dataset)
    loss = running_loss / total
    accuracy = (100.0 * running_correct) / total

    if epoch % 10 == 0:
        print('epoch %d: \t%s loss is \t%7.5f ;\t%s accuracy is \t%d/%d \t%7.3f%%' % (
            epoch, phase, loss, phase, running_correct, total, accuracy))

    return loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='CCNet Training')
    parser.add_argument('--config', type=str, default='./config/config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigParser(args.config)
    print(f"Using config: {args.config}")
    print(config)
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = config.training.gpu_ids

    print('weight of cross:', config.training.ce_weight)
    print('weight of contra:', config.training.contrastive_weight)
    print('weight of competition:', config.model.competition_weight)
    print('temperature:', config.training.temperature)

    # Create directories
    if not os.path.exists(config.training.checkpoint_path):
        os.makedirs(config.training.checkpoint_path)
    if not os.path.exists(config.training.results_path):
        os.makedirs(config.training.results_path)

    # Create dataset
    trainset = MyDataset(
        txt=config.dataset.train_set_file,
        transforms=None,
        train=True,
        imside=config.dataset.height,
        outchannels=config.dataset.channels
    )
    testset = MyDataset(
        txt=config.dataset.test_set_file,
        transforms=None,
        train=False,
        imside=config.dataset.height,
        outchannels=config.dataset.channels
    )

    data_loader_train = DataLoader(
        dataset=trainset,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
        shuffle=True
    )
    data_loader_test = DataLoader(
        dataset=testset,
        batch_size=128,
        num_workers=config.training.num_workers,
        shuffle=True
    )

    print('%s' % (time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())))
    print('------Init Model------')
    
    # Create model
    net = ccnet(
        num_classes=config.model.num_classes,
        weight=config.model.competition_weight
    )
    best_net = ccnet(
        num_classes=config.model.num_classes,
        weight=config.model.competition_weight
    )
    net.cuda()

    # Loss functions
    criterion = nn.CrossEntropyLoss()
    con_criterion = SupConLoss(
        temperature=config.training.temperature,
        base_temperature=config.training.temperature
    )

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=config.training.learning_rate)
    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=config.training.scheduler_step_size,
        gamma=config.training.scheduler_gamma
    )

    train_losses, train_accuracy = [], []
    val_losses, val_accuracy = [], []
    bestacc = 0

    # Training loop
    for epoch in range(config.training.num_epochs):
        epoch_loss, epoch_accuracy = fit(
            epoch, net, data_loader_train, 'training',
            optimizer, criterion, con_criterion, config
        )
        
        val_epoch_loss, val_epoch_accuracy = fit(
            epoch, net, data_loader_train, 'testing',
            optimizer, criterion, con_criterion, config
        )

        scheduler.step()

        # Logs
        train_losses.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        val_losses.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)

        # Save best model
        if epoch_accuracy >= bestacc:
            bestacc = epoch_accuracy
            torch.save(net.state_dict(), str(config.training.checkpoint_path) + '/net_params_best.pth')
            best_net = copy.deepcopy(net)

        # Save current model
        if epoch % 10 == 0 or epoch == (config.training.num_epochs - 1) and epoch != 0:
            torch.save(net.state_dict(), str(config.training.checkpoint_path) + '/net_params.pth')
            saveLossACC(train_losses, val_losses, train_accuracy, val_accuracy, bestacc, config.training.results_path)

        if epoch % config.training.save_interval == 0:
            torch.save(net.state_dict(), str(config.training.checkpoint_path) + f'/epoch_{epoch}_net_params.pth')

        if epoch % config.training.test_interval == 0 and epoch != 0:
            print('------------\n')
            test(net, config)

    print('------------\n')
    print('Last')
    test(net, config)

    print('------------\n')
    print('Best')
    test(best_net, config)


if __name__ == "__main__":
    main()