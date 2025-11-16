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
        
        # Parse Dataset section
        if 'Dataset' in self.config_dict:
            dataset_dict = self.config_dict['Dataset'].copy()
            dataset_dict.pop('config_file', None)
            self.dataset = Dataset(**dataset_dict)
        
        # Parse Model section
        if 'Model' in self.config_dict:
            model_dict = self.config_dict['Model'].copy()
            model_dict.pop('config_file', None)
            if 'use_pretrained' not in model_dict:
                model_dict['use_pretrained'] = False
            if 'pretrained_path' not in model_dict:
                model_dict['pretrained_path'] = None
            # 프로젝션 헤드 기본값 설정
            if 'use_projection' not in model_dict:
                model_dict['use_projection'] = True
            if 'projection_dim' not in model_dict:
                model_dict['projection_dim'] = 128
            self.model = Model(**model_dict)
        
        # Parse Training section
        if 'Training' in self.config_dict:
            training_dict = self.config_dict['Training'].copy()
            training_dict.pop('config_file', None)
            if 'batch_size' not in training_dict:
                training_dict['batch_size'] = 128
                
            #  ProxyAnchorLoss 기본값
            if 'use_proxy_anchor' not in training_dict:
                training_dict['use_proxy_anchor'] = True
            if 'proxy_margin' not in training_dict:
                training_dict['proxy_margin'] = 0.1
            if 'proxy_alpha' not in training_dict:
                training_dict['proxy_alpha'] = 32
            if 'proxy_lr_ratio' not in training_dict:
                training_dict['proxy_lr_ratio'] = 10
            if 'proxy_lambda' not in training_dict:
                training_dict['proxy_lambda'] = 0.3  #  고정 가중치 기본값
                
            self.training = Training(**training_dict)
            
            #  ProxyAnchorLoss 설정 출력
            if self.training.use_proxy_anchor:
                print(f" ProxyAnchorLoss configuration loaded:")
                print(f"   Margin (δ): {self.training.proxy_margin}")
                print(f"   Alpha (α): {self.training.proxy_alpha}")
                print(f"   LR ratio: {self.training.proxy_lr_ratio}x")
                print(f"   Lambda (fixed): {self.training.proxy_lambda}")  #  고정값
        
        # Parse Openset section
        if 'Openset' in self.config_dict:
            openset_dict = self.config_dict['Openset'].copy()
            openset_dict.pop('config_file', None)
            
            #  기본값 추가
            if 'threshold_mode' not in openset_dict:
                openset_dict['threshold_mode'] = 'far'
            if 'target_far' not in openset_dict:
                openset_dict['target_far'] = 0.01
            if 'verbose_calibration' not in openset_dict:
                openset_dict['verbose_calibration'] = True
            
            #  TTA 기본값 추가
            if 'tta_n_views' not in openset_dict:
                openset_dict['tta_n_views'] = 1  # 기본: 비활성화
            if 'tta_include_original' not in openset_dict:
                openset_dict['tta_include_original'] = True
            if 'tta_agree_k' not in openset_dict:
                openset_dict['tta_agree_k'] = 0
            if 'tta_augmentation_strength' not in openset_dict:
                openset_dict['tta_augmentation_strength'] = 0.5
            if 'tta_aggregation' not in openset_dict:
                openset_dict['tta_aggregation'] = 'median'
            
            #  기존 TTA 반복 기본값
            # if 'tta_n_repeats' not in openset_dict:
            #     openset_dict['tta_n_repeats'] = 1
            
            #  TTA 반복 기본값 추가
            if 'tta_n_repeats' not in openset_dict:
                openset_dict['tta_n_repeats'] = 1
            if 'tta_repeat_aggregation' not in openset_dict:
                openset_dict['tta_repeat_aggregation'] = 'median'
            if 'tta_verbose' not in openset_dict:
                openset_dict['tta_verbose'] = False
            
            #  타입별 TTA 반복 기본값 추가
            if 'tta_n_repeats_genuine' not in openset_dict:
                openset_dict['tta_n_repeats_genuine'] = None
            if 'tta_n_repeats_between' not in openset_dict:
                openset_dict['tta_n_repeats_between'] = None
            
            #  Impostor 비율 기본값 추가
            if 'impostor_ratio_between' not in openset_dict:
                openset_dict['impostor_ratio_between'] = 1.0
            if 'impostor_balance_total' not in openset_dict:
                openset_dict['impostor_balance_total'] = 6000
            
            self.openset = Openset(**openset_dict)
            
            #  모드 표시
            mode_info = f" (FAR {self.openset.target_far*100:.1f}%)" if self.openset.threshold_mode == 'far' else " (EER)"
            print(f" Open-set configuration loaded{mode_info}")
            
            #  TTA 설정 출력
            if openset_dict.get('tta_n_views', 1) > 1:
                print(f" TTA enabled: {openset_dict['tta_n_views']} views")
                print(f"   Include original: {openset_dict.get('tta_include_original', True)}")
                print(f"   Augmentation strength: {openset_dict.get('tta_augmentation_strength', 0.5)}")
                print(f"   Aggregation: {openset_dict.get('tta_aggregation', 'median')}")
                
                #  타입별 TTA 반복 설정 출력
                print(f"   Type-specific repeats:")
                print(f"     - Genuine: {self.openset.tta_n_repeats_genuine} repeats")
                print(f"     - Between: {self.openset.tta_n_repeats_between} repeats")
                print(f"   Repeat aggregation: {openset_dict.get('tta_repeat_aggregation', 'median')}")

                # 총 평가 횟수 계산
                total_evals_genuine = openset_dict['tta_n_views'] * self.openset.tta_n_repeats_genuine
                total_evals_between = openset_dict['tta_n_views'] * self.openset.tta_n_repeats_between
                
                print(f"   Total evaluations per sample:")
                print(f"     - Genuine: {total_evals_genuine}")
                print(f"     - Between: {total_evals_between}")
            
            #  Impostor 비율 출력
            print(f" Impostor score settings:")
            print(f"   Between impostor: {self.openset.impostor_ratio_between*100:.0f}%")
            print(f"   Total samples for balancing: {self.openset.impostor_balance_total}")

            #  실제 샘플 수 계산 및 출력
            n_between = int(self.openset.impostor_balance_total * self.openset.impostor_ratio_between)
            print(f"   Expected between impostor samples: {n_between}")
            
        else:
            self.openset = Openset(enabled=False)
            print(" Open-set configuration not found, using defaults (disabled)")
        
        # Parse Negative section
        if 'Negative' in self.config_dict:
            negative_dict = self.config_dict['Negative'].copy()
            negative_dict.pop('config_file', None)
            if 'base_id' not in negative_dict:
                negative_dict['base_id'] = 1
            self.negative = Negative(**negative_dict)
            print(" Negative configuration loaded")
        else:
            self.negative = Negative()
            print(" Negative configuration not found, using defaults")
    
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