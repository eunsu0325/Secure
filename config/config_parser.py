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
        
        # Add config file path
        for config_type in self.config_dict.values():
            config_type['config_file'] = self.filename
        
        # Convert paths to absolute paths
        for config_type in self.config_dict.values():
            for key, value in config_type.items():
                if key.endswith('_path') or key.endswith('_file'):
                    # 🔥 None 체크 추가 (pretrained_path가 없을 수 있음)
                    if value is not None:
                        config_type[key] = Path(value).absolute()
                    else:
                        config_type[key] = None
        
        # Parse sections
        if 'Dataset' in self.config_dict:
            self.dataset = Dataset(**self.config_dict['Dataset'])
        
        if 'Model' in self.config_dict:
            # 🔥 사전훈련 관련 기본값 설정
            model_dict = self.config_dict['Model']
            if 'use_pretrained' not in model_dict:
                model_dict['use_pretrained'] = False
            if 'pretrained_path' not in model_dict:
                model_dict['pretrained_path'] = None
            
            self.model = Model(**model_dict)
        
        if 'Training' in self.config_dict:
            self.training = Training(**self.config_dict['Training'])
    
    def get_config(self):
        """Config 객체들을 포함한 namespace 반환"""
        import types
        config = types.SimpleNamespace()
        config.dataset = self.dataset
        config.model = self.model
        config.training = self.training
        return config
    
    def __str__(self):
        string = ''
        for config_type_name, config_type in self.config_dict.items():
            string += f'----- {config_type_name} --- START -----\n'
            for name, value in config_type.items():
                if name != 'config_file':
                    string += f'{name:25} : {value}\n'
            string += f'----- {config_type_name} --- END -------\n'
        return string