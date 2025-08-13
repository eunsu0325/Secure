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
            self.training = Training(**training_dict)
        
        if 'Openset' in self.config_dict:
            openset_dict = self.config_dict['Openset'].copy()
            openset_dict.pop('config_file', None)
            self.openset = Openset(**openset_dict)
            print("🐋 Open-set configuration loaded")
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