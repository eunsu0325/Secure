# config/config_parser.py

import dataclasses
import sys
from pathlib import Path
from typing import Union, List
import yaml
from config.config import Dataset, Model, Training


class ConfigParser:
    def __init__(self, config_file: Union[str, Path]) -> None:
        self.filename = Path(config_file)
        self.config_dict = {}
        self.dataset = None
        self.model = None
        self.training = None
        self.parse()
    
    def parse(self):
        with open(self.filename, 'r', encoding='utf-8') as file:
            self.config_dict = yaml.safe_load(file)
        
        # Convert lists to tuples
        for config_type in self.config_dict.values():
            for key, value in config_type.items():
                if isinstance(value, List):
                    config_type[key] = tuple(value)
        
        # 🌈 config_file 추가 부분 제거 (에러 원인)
        # 각 데이터클래스가 config_file 필드를 갖지 않으므로 제거
        # for config_type in self.config_dict.values():
        #     config_type['config_file'] = self.filename
        
        # Convert paths to absolute paths
        for config_type in self.config_dict.values():
            for key, value in config_type.items():
                if key.endswith('_path') or key.endswith('_file'):
                    # None 체크 추가 (pretrained_path가 없을 수 있음)
                    if value is not None:
                        config_type[key] = Path(value).absolute()
                    else:
                        config_type[key] = None
        
        # Parse sections
        if 'Dataset' in self.config_dict:
            # 🌈 안전하게 복사 후 config_file 제거 (혹시 있다면)
            dataset_dict = self.config_dict['Dataset'].copy()
            dataset_dict.pop('config_file', None)
            self.dataset = Dataset(**dataset_dict)
        
        if 'Model' in self.config_dict:
            # 🌈 안전하게 복사 후 config_file 제거 (혹시 있다면)
            model_dict = self.config_dict['Model'].copy()
            model_dict.pop('config_file', None)
            
            # 사전훈련 관련 기본값 설정
            if 'use_pretrained' not in model_dict:
                model_dict['use_pretrained'] = False
            if 'pretrained_path' not in model_dict:
                model_dict['pretrained_path'] = None
            
            self.model = Model(**model_dict)
        
        if 'Training' in self.config_dict:
            # 🌈 안전하게 복사 후 config_file 제거 (혹시 있다면)
            training_dict = self.config_dict['Training'].copy()
            training_dict.pop('config_file', None)
            self.training = Training(**training_dict)
    
    def get_config(self):
        """Config 객체들을 포함한 namespace 반환"""
        import types
        config = types.SimpleNamespace()
        config.dataset = self.dataset
        config.model = self.model
        config.training = self.training
        # 🌈 config 파일 경로도 추가 (필요시 접근 가능)
        config.config_file = self.filename
        return config
    
    def __str__(self):
        string = ''
        for config_type_name, config_type in self.config_dict.items():
            string += f'----- {config_type_name} --- START -----\n'
            for name, value in config_type.items():
                # config_file은 출력에서 제외
                if name != 'config_file':
                    string += f'{name:25} : {value}\n'
            string += f'----- {config_type_name} --- END -------\n'
        return string
    
    def __repr__(self):
        return self.__str__()