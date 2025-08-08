import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torch.optim import lr_scheduler
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from PIL import Image

from loss import SupConLoss
from models import get_scr_transforms
from utils.util import AverageMeter


class MemoryDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform, train=True):
        self.paths = paths
        
        # 텐서인 경우 CPU로 이동 후 numpy 변환
        if torch.is_tensor(labels):
            self.labels = labels.cpu().numpy()
        else:
            self.labels = np.array(labels)
            
        self.transform = transform
        self.train = train
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        
        # 이미지 로드
        img = Image.open(path).convert('L')
        
        # 같은 이미지에 다른 augmentation 적용
        data1 = self.transform(img)
        data2 = self.transform(img)
        
        return [data1, data2], label


class SCRTrainer:
    """
    Supervised Contrastive Replay Trainer.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 ncm_classifier,
                 memory_buffer,
                 config,
                 device='cuda'):
        
        self.model = model
        self.ncm = ncm_classifier
        self.memory_buffer = memory_buffer
        self.config = config
        self.device = device
        
        # Loss function
        self.criterion = SupConLoss(
            temperature=config.training.temperature,
            base_temperature=config.training.temperature
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.training.learning_rate
        )
        
        # Scheduler
        self.scheduler = lr_scheduler.StepLR(
            self.optimizer,
            step_size=config.training.scheduler_step_size,
            gamma=config.training.scheduler_gamma
        )
        
        # Transform
        self.train_transform = get_scr_transforms(
            train=True,
            imside=config.dataset.height,
            channels=config.dataset.channels
        )
        
        self.test_transform = get_scr_transforms(
            train=False,
            imside=config.dataset.height,
            channels=config.dataset.channels
        )
        
        # 전체 메모리 데이터셋 캐시 (효율성 개선)
        self._full_memory_dataset = None
        self._memory_size_at_last_update = 0
        
        # Statistics
        self.experience_count = 0
        
    def train_experience(self, user_id: int, image_paths: List[str], labels: List[int]) -> Dict:
        """하나의 experience (한 명의 사용자) 학습."""
        
        print(f"\n=== Training Experience {self.experience_count}: User {user_id} ===")
        
        # 👻 원본 labels를 보존 (중요!)
        original_labels = labels.copy()
        
        # 현재 사용자 데이터셋 생성
        current_dataset = MemoryDataset(
            paths=image_paths,
            labels=labels,
            transform=self.train_transform,
            train=True
        )
        
        # 학습 통계
        loss_avg = AverageMeter('Loss')
        
        # SCR 논문 방식: epoch당 여러 iteration
        self.model.train()
        
        for epoch in range(self.config.training.scr_epochs):
            epoch_loss = 0
            
            for iteration in range(self.config.training.iterations_per_epoch):
                
                # 1. 현재 데이터에서 B_n 샘플링 (중복 허용)
                current_indices = np.random.choice(
                    len(current_dataset), 
                    size=self.config.training.scr_batch_size,
                    replace=True
                )
                current_subset = Subset(current_dataset, current_indices)
                
                # 2. 메모리에서 B_M 샘플링
                if len(self.memory_buffer) > 0:
                    # 메모리에서 샘플링
                    memory_paths, memory_labels = self.memory_buffer.sample(
                        self.config.training.scr_batch_size
                    )
                    
                    if torch.is_tensor(memory_labels):
                        memory_labels = memory_labels.cpu().tolist()

                    # 메모리 데이터셋 생성
                    memory_dataset = MemoryDataset(
                        paths=memory_paths,
                        labels=memory_labels,
                        transform=self.train_transform,
                        train=True
                    )
                    
                    # 3. ConcatDataset으로 결합
                    combined_dataset = ConcatDataset([current_subset, memory_dataset])
                else:
                    combined_dataset = current_subset
                
                # 4. DataLoader로 배치 생성
                batch_loader = DataLoader(
                    combined_dataset,
                    batch_size=len(combined_dataset),
                    shuffle=False,
                    num_workers=0
                )
                
                # 5. 학습
                # 💀 기존: for data, labels in batch_loader:
                for data, batch_labels in batch_loader:  # 👻 변수명 변경
                    batch_size = len(batch_labels)  # 👻 변경
                    
                    # Flatten positive pairs
                    data_views = []
                    for i in range(batch_size):
                        data_views.append(data[0][i])
                        data_views.append(data[1][i])
                    
                    data = torch.stack(data_views).to(self.device)
                    batch_labels = batch_labels.to(self.device)  # 👻 변경
                    
                    # Forward
                    self.optimizer.zero_grad()
                    features = self.model(data)
                    
                    # Reshape for SupConLoss
                    f1, f2 = torch.split(features, [batch_size, batch_size], dim=0)
                    features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                    
                    # Calculate loss
                    loss = self.criterion(features, batch_labels)  # 👻 변경
                    loss_avg.update(loss.item(), batch_size)
                    
                    # Backward with gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / self.config.training.iterations_per_epoch
                print(f"  Epoch [{epoch+1}/{self.config.training.scr_epochs}] Loss: {avg_loss:.4f}")
        
        # 5. NCM 업데이트 (메모리 버퍼 데이터만 사용)
        self._update_ncm()
        
        # 6. 메모리 버퍼 업데이트
        print(f"\n=== Before buffer update ===")
        print(f"Buffer size: {len(self.memory_buffer)}")
        print(f"Buffer seen classes: {self.memory_buffer.seen_classes if hasattr(self.memory_buffer, 'seen_classes') else 'N/A'}")
        print(f"image_paths: {type(image_paths)}, len: {len(image_paths)}")
        # 💀 print(f"labels: {type(labels)}, len: {len(labels)}")
        print(f"original_labels: {type(original_labels)}, len: {len(original_labels)}")  # 👻
        # 💀 print(f"labels content: {labels}")
        print(f"original_labels content: {original_labels}")  # 👻
        
        # 💀 self.memory_buffer.update_from_dataset(image_paths, labels)
        self.memory_buffer.update_from_dataset(image_paths, original_labels)  # 👻 원본 labels 사용
        self._full_memory_dataset = None  # 캐시 무효화
        print(f"Memory buffer size: {len(self.memory_buffer)}")
        
        # 7. Experience 카운터 증가
        self.experience_count += 1
        
        # 8. Scheduler step
        self.scheduler.step()
        
        return {
            'experience': self.experience_count - 1,
            'user_id': user_id,
            'loss': loss_avg.avg,
            'memory_size': len(self.memory_buffer)
        }
    
    @torch.no_grad()
    def _update_ncm(self):
        """
        NCM classifier의 class means를 업데이트합니다.
        메모리 버퍼의 데이터만 사용합니다.
        """
        if len(self.memory_buffer) == 0:
            return
            
        self.model.eval()
        
        # 메모리 버퍼에서 모든 데이터 가져오기
        all_paths, all_labels = self.memory_buffer.get_all_data()
        
        # 데이터셋 생성
        dataset = MemoryDataset(
            paths=all_paths,
            labels=all_labels,
            transform=self.test_transform,
            train=False
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=self.config.training.num_workers
        )
        
        # 클래스별로 features 수집
        class_features = {}
        
        for data, labels in dataloader:
            # data1만 사용 (테스트 시)
            data = data[0].to(self.device)
            labels = labels.to(self.device)
            
            # Feature extraction
            features = self.model.getFeatureCode(data)
            
            # 클래스별로 분류
            for i, label in enumerate(labels):
                label_item = label.item()
                if label_item not in class_features:
                    class_features[label_item] = []
                class_features[label_item].append(features[i].cpu())
        
        # 클래스별 평균 계산
        class_means = {}
        for label, features_list in class_features.items():
            if len(features_list) > 0:
                mean_feature = torch.stack(features_list).mean(dim=0)
                mean_feature = mean_feature / mean_feature.norm()
                class_means[label] = mean_feature
        
        # NCM 업데이트
        self.ncm.update_class_means_dict(
            class_means, 
            momentum=self.config.training.ncm_momentum
        )
        
        print(f"Updated NCM with {len(class_means)} classes")
        
        self.model.train()
    
    def evaluate(self, test_dataset: Dataset) -> float:
        """
        NCM을 사용하여 정확도를 평가합니다.
        """
        self.model.eval()
        
        dataloader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=self.config.training.num_workers
        )
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in dataloader:
                # data1만 사용 (테스트 시)
                data = data[0].to(self.device)
                labels = labels.to(self.device)
                
                # Feature extraction
                features = self.model.getFeatureCode(data)
                
                # NCM prediction
                predictions = self.ncm.predict(features)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100.0 * correct / total
        return accuracy
    
    def save_checkpoint(self, path: str):
        """체크포인트 저장"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ncm_state_dict': self.ncm.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'experience_count': self.experience_count,
            'memory_buffer_size': len(self.memory_buffer)
        }, path)
    
    def load_checkpoint(self, path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ncm.load_state_dict(checkpoint['ncm_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.experience_count = checkpoint['experience_count']