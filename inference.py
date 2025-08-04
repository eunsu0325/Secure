import os
import argparse
import time
import sys

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import transforms
from torchvision import models

import pickle
import numpy as np
from PIL import Image
import cv2 as cv
import faiss  # 추가
from loss import SupConLoss

import matplotlib.pyplot as plt
from utils.util import plotLossACC, saveLossACC, saveGaborFilters, saveParameters, saveFeatureMaps

plt.switch_backend('agg')

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

    batch_size = 1024  # 128

    data_loader_train = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=0)
    data_loader_test = DataLoader(dataset=testset, batch_size=batch_size, num_workers=0)

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
    assert num_training_samples % classNumel == 0
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

    print('completed feature extraction for test set.')
    print('featDB_test.shape: ', featDB_test.shape)
    print('\nfeature extraction done!')
    print('\n\n')
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
    
    # rank-1 acc with Faiss (더 빠른 버전)
    # Faiss index 생성
    feature_dim = featDB_train.shape[1]
    index = faiss.IndexFlatL2(feature_dim)
    index.add(featDB_train.astype(np.float32))
    
    # 각 테스트 샘플에 대해 가장 가까운 train 샘플 찾기
    distances, indices = index.search(featDB_test.astype(np.float32), k=1)
    
    # 정확도 계산
    corr = 0
    for i in range(ntest):
        probeID = iddb_test[i]
        galleryID = iddb_train[indices[i][0]]
        
        if probeID == galleryID:
            corr += 1
        else:
            # 잘못 매칭된 샘플 저장
            testname = fileDB_test[i]
            trainname = fileDB_train[indices[i][0]]
            im_test = cv.imread(testname)
            im_train = cv.imread(trainname)
            if im_test is not None and im_train is not None:
                img = np.concatenate((im_test, im_train), axis=1)
                cv.imwrite(veriEER_path + '/rank1_hard/%6.4f_%s_%s.png' % (
                    distances[i][0], testname[-13:-4], trainname[-13:-4]), img)
    
    # 기존 방식 (주석처리) 🔥
    # cnt = 0
    # corr = 0
    # for i in range(ntest):
    #     probeID = iddb_test[i]
    #     dis = np.zeros((ntrain, 1))
    #     for j in range(ntrain):
    #         dis[j] = s[cnt]
    #         cnt += 1
    #     idx = np.argmin(dis[:])
    #     galleryID = iddb_train[idx]
    #
    #     if probeID == galleryID:
    #         corr += 1
    #     else:
    #         testname = fileDB_test[i]
    #         trainname = fileDB_train[idx]
    #         im_test = cv.imread(testname)
    #         im_train = cv.imread(trainname)
    #         img = np.concatenate((im_test, im_train), axis=1)
    #         cv.imwrite(veriEER_path + '/rank1_hard/%6.4f_%s_%s.png' % (
    #             np.min(dis[:]), testname[-13:-4], trainname[-13:-4]), img)

    rankacc = corr / ntest * 100
    print('rank-1 acc: %.3f%%' % rankacc)
    print('-----------')

    with open(veriEER_path + '/rank1.txt', 'w') as f:
        f.write('rank-1 acc: %.3f%%' % rankacc)


def main():
    parser = argparse.ArgumentParser(description="CCNet Inference")
    parser.add_argument('--config', type=str, default='./config/config.yaml',
                        help='Path to config file')
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint file")
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigParser(args.config)
    print(f"Using config: {args.config}")
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = config.training.gpu_ids

    # Create model
    net = ccnet(
        # num_classes=config.model.num_classes,  # 🔥 제거
        weight=config.model.competition_weight
    )
    net.load_state_dict(torch.load(args.checkpoint), strict=False)
    
    # Run test
    test(net, config)


if __name__ == "__main__":
    main()