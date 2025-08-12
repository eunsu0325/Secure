#models/dataset.py
# -*- coding:utf-8 -*-
import os
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torchvision import transforms as T


class NormSingleROI(object):
    """
    Normalize the input image (exclude the black region) with 0 mean and 1 std.
    [c,h,w]
    """
    def __init__(self, outchannels=1):
        self.outchannels = outchannels

    def __call__(self, tensor):
        c,h,w = tensor.size()
   
        if c != 1:
            raise TypeError('only support graysclae image.')

        tensor = tensor.view(c, h*w)
        idx = tensor > 0
        t = tensor[idx]

        m = t.mean()
        s = t.std() 
        t = t.sub_(m).div_(s+1e-6)
        tensor[idx] = t
        
        tensor = tensor.view(c, h, w)

        if self.outchannels > 1:
            tensor = torch.repeat_interleave(tensor, repeats = self.outchannels, dim = 0)
    
        return tensor


def get_scr_transforms(train=True, imside=128, channels=1):
    """SCR을 위한 transform만 반환"""
    if not train:
        return T.Compose([
            T.Resize(imside),
            T.ToTensor(),
            NormSingleROI(outchannels=channels)
        ])
    else:
        return T.Compose([
            T.Resize(imside),
            T.RandomChoice(transforms=[
                T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                T.RandomResizedCrop(size=imside, scale=(0.8,1.0), ratio=(1.0, 1.0)),
                T.RandomPerspective(distortion_scale=0.15, p=1),
                T.RandomChoice(transforms=[
                    T.RandomRotation(degrees=10, interpolation=Image.BICUBIC, expand=False, center=(0.5*imside, 0.0)),
                    T.RandomRotation(degrees=10, interpolation=Image.BICUBIC, expand=False, center=(0.0, 0.5*imside)),
                ]),
            ]),
            T.ToTensor(),
            NormSingleROI(outchannels=channels)
        ])


class MyDataset(data.Dataset):
    '''
    Load and process the ROI images::

    INPUT::
    txt: a text file containing pathes & labels of the input images \n
    transforms: None 
    train: True for a training set, and False for a testing set
    imside: the image size of the output image [imside x imside]
    outchannels: 1 for grayscale image, and 3 for RGB image
    dual_views: True면 2뷰 생성, False면 1뷰만 생성

    OUTPUT::
    [batch, outchannels, imside, imside]
    '''
    
    # 🌈 dual_views 파라미터 추가
    def __init__(self, txt, transforms=None, train=True, imside=128, outchannels=1, dual_views=None):        

        self.train = train
        self.imside = imside # 128, 224
        self.chs = outchannels # 1, 3
        self.text_path = txt        
        self.transforms = transforms
        
        # 🌈 dual_views 설정: 명시적 지정 없으면 train 값 따라감
        self.dual_views = dual_views if dual_views is not None else train

        if transforms is None:
            if not train: 
                self.transforms = T.Compose([                                            
                    T.Resize(self.imside),                  
                    T.ToTensor(),   
                    NormSingleROI(outchannels=self.chs)
                ]) 
            else:
                self.transforms = T.Compose([                  
                    T.Resize(self.imside),
                    T.RandomChoice(transforms=[
                        T.ColorJitter(brightness=0, contrast=0.05, saturation=0, hue=0),
                        T.RandomResizedCrop(size=self.imside, scale=(0.8,1.0), ratio=(1.0, 1.0)),
                        T.RandomPerspective(distortion_scale=0.15, p=1),
                        T.RandomChoice(transforms=[
                            T.RandomRotation(degrees=10, interpolation=Image.BICUBIC, expand=False, center=(0.5*self.imside, 0.0)),
                            T.RandomRotation(degrees=10, interpolation=Image.BICUBIC, expand=False, center=(0.0, 0.5*self.imside)),
                        ]),
                    ]),     
                    T.ToTensor(),
                    NormSingleROI(outchannels=self.chs)                   
                ])

        self._read_txt_file()

    def _read_txt_file(self):
        self.images_path = []
        self.images_label = []
        # 🌈 클래스별 인덱스 캐싱 추가
        self.class_to_indices = {}

        txt_file = self.text_path

        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for idx, line in enumerate(lines):
                item = line.strip().split(' ')
                self.images_path.append(item[0])
                # 🌈 바로 int로 변환
                label = int(item[1])
                self.images_label.append(label)
                
                # 🌈 클래스별 인덱스 캐싱
                if label not in self.class_to_indices:
                    self.class_to_indices[label] = []
                self.class_to_indices[label].append(idx)

    def __getitem__(self, index):
        img_path = self.images_path[index]
        label = self.images_label[index]  # 🌈 이미 int
        
        # 🌈 조건부 뷰 생성
        if self.dual_views:
            # 🌈 방법 1: 같은 이미지의 다른 증강 (MemoryDataset과 통일) - 권장
            img = Image.open(img_path).convert('L')
            data1 = self.transforms(img)
            data2 = self.transforms(img)
            data = [data1, data2]
            
            # 😶‍🌫️ 방법 2: 같은 클래스의 다른 샘플 (기존 방식) - 필요시 주석 해제
            # # 🌈 캐시된 인덱스 사용 (빠름!)
            # same_class_indices = self.class_to_indices[label]
            # 
            # if self.train and len(same_class_indices) > 1:
            #     idx2 = index
            #     while idx2 == index:
            #         idx2 = np.random.choice(same_class_indices)
            # else:
            #     idx2 = index
            # 
            # img_path2 = self.images_path[idx2]
            # 
            # data1 = Image.open(img_path).convert('L')
            # data1 = self.transforms(data1)
            # 
            # data2 = Image.open(img_path2).convert('L')
            # data2 = self.transforms(data2)
            # 
            # data = [data1, data2]
            
        else:
            # 🌈 1뷰만 생성 (평가용)
            img = Image.open(img_path).convert('L')
            data = self.transforms(img)
            # 단일 텐서 반환 (리스트 아님)
        
        return data, label  # 🌈 int() 제거

    def __len__(self):
        return len(self.images_path)