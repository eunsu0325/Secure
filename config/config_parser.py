# config/config_parser.py
import dataclasses
import sys
from pathlib import Path
from typing import Union, List
import yaml
from config.config import Dataset, Model, Training, Openset, Negative


class ConfigParser:
    def __init__(self, config_file: Union[str, Path]) -> None:
        self.filename = Path(config_file)
        self.config_dict = {}
        self.dataset = None
        self.model = None
        self.training = None
        self.openset = None
        self.negative = None
        self.parse()
    
    def parse(self):
        with open(self.filename, 'r', encoding='utf-8') as file:
            self.config_dict = yaml.safe_load(file) or {}
        
        if not isinstance(self.config_dict, dict):
            raise ValueError(f"Invalid config format in {self.filename}")
        
        # Convert lists to tuples
        for config_type in self.config_dict.values():
            for key, value in config_type.items():
                if isinstance(value, list):
                    config_type[key] = tuple(value)
        
        # Convert paths to absolute paths
        for config_type in self.config_dict.values():
            for key, value in config_type.items():
                if key.endswith('_path') or key.endswith('_file'):
                    if value is not None:
                        config_type[key] = Path(value).absolute()
                    else:
                        config_type[key] = None
        
        # Parse sections
        if 'Dataset' in self.config_dict:
            dataset_dict = self.config_dict['Dataset'].copy()
            dataset_dict.pop('config_file', None)
            self.dataset = Dataset(**dataset_dict)
        
        if 'Model' in self.config_dict:
            model_dict = self.config_dict['Model'].copy()
            model_dict.pop('config_file', None)
            if 'use_pretrained' not in model_dict:
                model_dict['use_pretrained'] = False
            if 'pretrained_path' not in model_dict:
                model_dict['pretrained_path'] = None
            # 🧀 프로젝션 헤드 기본값 설정
            if 'use_projection' not in model_dict:
                model_dict['use_projection'] = True
            if 'projection_dim' not in model_dict:
                model_dict['projection_dim'] = 128
            self.model = Model(**model_dict)
        
        if 'Training' in self.config_dict:
            training_dict = self.config_dict['Training'].copy()
            training_dict.pop('config_file', None)
            if 'batch_size' not in training_dict:
                training_dict['batch_size'] = 128
                
            # 💎 ProxyLoss 기본값 설정
            if 'use_proxy_loss' not in training_dict:
                training_dict['use_proxy_loss'] = True
            if 'proxy_lambda' not in training_dict:
                training_dict['proxy_lambda'] = 0.3
            if 'proxy_temperature' not in training_dict:
                training_dict['proxy_temperature'] = 0.15
            if 'proxy_topk' not in training_dict:
                training_dict['proxy_topk'] = 30
            if 'proxy_full_until' not in training_dict:
                training_dict['proxy_full_until'] = 100
            if 'proxy_warmup_classes' not in training_dict:
                training_dict['proxy_warmup_classes'] = 5
                
            # 💎 Lambda 스케줄 파싱
            if 'proxy_lambda_schedule' in training_dict:
                # YAML에서 dict로 파싱됨
                pass
            else:
                # 기본 스케줄 설정 (Training.__post_init__에서 처리)
                training_dict['proxy_lambda_schedule'] = None
            
            # 💎 커버리지 샘플링 기본값
            if 'use_coverage_sampling' not in training_dict:
                training_dict['use_coverage_sampling'] = True
            if 'coverage_k_per_class' not in training_dict:
                training_dict['coverage_k_per_class'] = 2
            
            # 💎 프로토타입 설정
            if 'prototype_beta' not in training_dict:
                training_dict['prototype_beta'] = 0.05
                
            self.training = Training(**training_dict)
            
            # 💎 ProxyLoss 설정 출력
            if self.training.use_proxy_loss:
                print(f"💎 ProxyContrastLoss configuration loaded:")
                print(f"   Lambda: {self.training.proxy_lambda}")
                print(f"   Temperature: {self.training.proxy_temperature}")
                print(f"   Schedule: {self.training.proxy_lambda_schedule}")
        
        if 'Openset' in self.config_dict:
            openset_dict = self.config_dict['Openset'].copy()
            openset_dict.pop('config_file', None)
            
            # 🍎 FAR 모드 기본값 추가 (3줄만!)
            if 'threshold_mode' not in openset_dict:
                openset_dict['threshold_mode'] = 'far'
            if 'target_far' not in openset_dict:
                openset_dict['target_far'] = 0.01
            if 'verbose_calibration' not in openset_dict:
                openset_dict['verbose_calibration'] = True
            
            self.openset = Openset(**openset_dict)
            
            # 🍎 모드 표시 (1줄 추가)
            mode_info = f" (FAR {self.openset.target_far*100:.1f}%)" if self.openset.threshold_mode == 'far' else " (EER)"
            print(f"🐋 Open-set configuration loaded{mode_info}")
        else:
            self.openset = Openset(enabled=False)
            print("📌 Open-set configuration not found, using defaults (disabled)")
        
        if 'Negative' in self.config_dict:
            negative_dict = self.config_dict['Negative'].copy()
            negative_dict.pop('config_file', None)
            if 'base_id' not in negative_dict:
                negative_dict['base_id'] = 1
            self.negative = Negative(**negative_dict)
            print("🔥 Negative configuration loaded")
        else:
            self.negative = Negative()
            print("📌 Negative configuration not found, using defaults")
    
    def get_config(self):
        """Config 객체들을 포함한 namespace 반환"""
        import types
        config = types.SimpleNamespace()
        config.dataset = self.dataset
        config.model = self.model
        config.training = self.training
        config.openset = self.openset
        config.negative = self.negative
        config.config_file = self.filename
        return config
    
    def __str__(self):
        string = ''
        for config_type_name, config_type in self.config_dict.items():
            string += f'----- {config_type_name} --- START -----\n'
            for name, value in config_type.items():
                if name != 'config_file':
                    string += f'{name:25} : {value}\n'
            string += f'----- {config_type_name} --- END -------\n'
        
        if self.openset and 'Openset' not in self.config_dict:
            string += f'----- Openset (Default) --- START -----\n'
            for field in dataclasses.fields(self.openset):
                value = getattr(self.openset, field.name)
                string += f'{field.name:25} : {value}\n'
            string += f'----- Openset (Default) --- END -------\n'
        
        if self.negative and 'Negative' not in self.config_dict:
            string += f'----- Negative (Default) --- START -----\n'
            for field in dataclasses.fields(self.negative):
                value = getattr(self.negative, field.name)
                string += f'{field.name:25} : {value}\n'
            string += f'----- Negative (Default) --- END -------\n'
        
        return string