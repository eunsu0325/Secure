# scr/data_stream.py
import os
from typing import List, Tuple, Dict
from collections import defaultdict


class ExperienceStream:
    """SCR을 위한 데이터 스트림 관리자."""
    
    def __init__(self, 
                 train_file: str,
                 negative_file: str,
                 num_negative_classes: int = 10):
        self.train_file = train_file
        self.negative_file = negative_file
        self.num_negative_classes = num_negative_classes
        
        # 데이터 로드
        try:
            self.user_data = self._load_data(train_file)
            self.negative_data = self._load_negative_data(negative_file)
        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")
        
        # Experience 매핑 (빠진 번호 대응)
        user_ids = sorted(self.user_data.keys())
        self.user_mapping = {i: uid for i, uid in enumerate(user_ids)}
        
        self.num_users = len(self.user_data)
        self.current_experience = 0
        
        print(f"Loaded {self.num_users} users (IDs: {user_ids[0]}-{user_ids[-1]})")
        print(f"Loaded {len(self.negative_data)} negative classes")
    
    def _load_data(self, txt_file: str) -> Dict[int, List[str]]:
        """txt 파일에서 데이터를 로드하고 사용자별로 정리합니다."""
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"File not found: {txt_file}")
            
        user_data = defaultdict(list)
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                try:
                    parts = line.strip().split(' ')
                    if len(parts) != 2:
                        print(f"Warning: Skipping invalid line {i}: {line.strip()}")
                        continue
                        
                    path, label = parts
                    user_id = int(label)
                    user_data[user_id].append(path)
                    
                except Exception as e:
                    print(f"Error processing line {i}: {e}")
                    continue
        
        return dict(user_data)
    
    def _load_negative_data(self, txt_file: str) -> Dict[int, List[str]]:
        """Negative 샘플을 로드합니다."""
        if not os.path.exists(txt_file):
            raise FileNotFoundError(f"File not found: {txt_file}")
            
        negative_data = defaultdict(list)
        
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                try:
                    parts = line.strip().split(' ')
                    if len(parts) != 2:
                        continue
                        
                    path, label = parts
                    class_id = int(label)
                    
                    if class_id < self.num_negative_classes:
                        negative_id = -class_id - 1  # 음수 ID 사용
                        negative_data[negative_id].append(path)
                        
                except Exception as e:
                    print(f"Error processing negative line {i}: {e}")
                    continue
        
        # 디버깅 메시지
        if negative_data:
            neg_ids = list(negative_data.keys())
            print(f"[DEBUG] Negative class IDs: {neg_ids} (using negative IDs)")
            print(f"[DEBUG] If NCM fails with negative IDs, consider using positive offset")
        
        return dict(negative_data)
    
    def get_experience(self, exp_id: int) -> Tuple[int, List[str], List[int]]:
        """특정 experience의 데이터를 반환합니다."""
        if exp_id >= self.num_users:
            raise ValueError(f"Experience {exp_id} out of range (max: {self.num_users-1})")
        
        user_id = self.user_mapping[exp_id]
        image_paths = self.user_data[user_id]
        labels = [user_id] * len(image_paths)
        
        return user_id, image_paths, labels
    
    def get_negative_samples(self) -> Tuple[List[str], List[int]]:
        """모든 negative 샘플을 반환합니다."""
        all_paths = []
        all_labels = []
        
        for neg_class_id, paths in self.negative_data.items():
            all_paths.extend(paths)
            all_labels.extend([neg_class_id] * len(paths))
        
        return all_paths, all_labels
    
    def __iter__(self):
        self.current_experience = 0
        return self
    
    def __next__(self):
        if self.current_experience >= self.num_users:
            raise StopIteration
        
        exp_data = self.get_experience(self.current_experience)
        self.current_experience += 1
        return exp_data
    
    def __len__(self):
        return self.num_users
    
    def get_statistics(self) -> Dict:
        """데이터셋 통계 정보"""
        stats = {
            'num_users': self.num_users,
            'num_negative_classes': len(self.negative_data),
            'samples_per_user': {},
            'total_samples': 0,
            'negative_samples': sum(len(paths) for paths in self.negative_data.values()),
            'missing_user_ids': [],
            'user_id_range': (min(self.user_data.keys()), max(self.user_data.keys()))
        }
        
        # 빠진 사용자 ID 찾기
        all_ids = set(range(stats['user_id_range'][0], stats['user_id_range'][1] + 1))
        existing_ids = set(self.user_data.keys())
        stats['missing_user_ids'] = sorted(all_ids - existing_ids)
        
        for user_id, paths in self.user_data.items():
            count = len(paths)
            if count not in stats['samples_per_user']:
                stats['samples_per_user'][count] = 0
            stats['samples_per_user'][count] += 1
            stats['total_samples'] += count
        
        return stats