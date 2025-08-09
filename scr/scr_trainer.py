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
from utils.pretrained_loader import PretrainedLoader


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
        
        # 사전훈련 로딩 (모델을 device로 옮기기 전에)
        if hasattr(config.model, 'use_pretrained') and config.model.use_pretrained:
            if config.model.pretrained_path and config.model.pretrained_path.exists():
                print(f"\n📦 Loading pretrained weights in SCRTrainer...")
                loader = PretrainedLoader()
                try:
                    model = loader.load_ccnet_pretrained(
                        model=model,
                        checkpoint_path=config.model.pretrained_path,
                        device=device,
                        verbose=True
                    )
                    print("✅ Pretrained weights loaded successfully in trainer!")
                except Exception as e:
                    print(f"⚠️  Failed to load pretrained: {e}")
                    print("Continuing with current weights...")
            else:
                print(f"⚠️  Pretrained path not found: {config.model.pretrained_path}")
        
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
        
        # 사전훈련 사용 여부 로그
        if hasattr(config.model, 'use_pretrained') and config.model.use_pretrained:
            print(f"🔥 SCRTrainer initialized with pretrained model")
        else:
            print(f"🎲 SCRTrainer initialized with random weights")
    
    def train_experience(self, user_id: int, image_paths: List[str], labels: List[int]) -> Dict:
        """하나의 experience (한 명의 사용자) 학습."""
        
        print(f"\n=== Training Experience {self.experience_count}: User {user_id} ===")
        
        # 원본 labels를 보존 (중요!)
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
                
                # 1. 현재 데이터에서 B_n 샘플링 (중복 제거 + 안전한 크기)
                current_indices = np.random.choice(
                    len(current_dataset), 
                    size=min(len(current_dataset), self.config.training.scr_batch_size),
                    replace=False  # 중복 제거!
                )
                current_subset = Subset(current_dataset, current_indices)
                
                # 2. 메모리에서 B_M 샘플링
                if len(self.memory_buffer) > 0:
                    # 메모리에서 샘플링
                    memory_paths, memory_labels = self.memory_buffer.sample(
                        self.config.training.memory_batch_size
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
                    shuffle=False,  # 순서 유지 중요!
                    num_workers=0
                )
                
                # 5. 학습
                for data, batch_labels in batch_loader:
                    batch_size = len(batch_labels)
                    
                    # 올바른 view 배치
                    view1 = data[0]  # 첫 번째 증강들
                    view2 = data[1]  # 두 번째 증강들
                    
                    # 올바른 순서로 연결
                    x = torch.cat([view1, view2], dim=0).to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    # Forward
                    self.optimizer.zero_grad()
                    features = self.model(x)  # [2*batch_size, feature_dim]
                    
                    # 올바른 페어링
                    f1 = features[:batch_size]
                    f2 = features[batch_size:]
                    
                    # 검증: 첫 번째 iteration에서 positive similarity 확인
                    if iteration == 0 and epoch == 0:
                        with torch.no_grad():
                            # Positive pairs: f1[i]와 f2[i]
                            pos_sim = torch.nn.functional.cosine_similarity(f1, f2, dim=1).mean()
                            # Negative pairs: f1[i]와 f2[i+1]
                            neg_sim = torch.nn.functional.cosine_similarity(
                                f1, f2.roll(shifts=1, dims=0), dim=1
                            ).mean()
                            print(f"  📊 Similarity check:")
                            print(f"     Positive pairs: {pos_sim:.4f} (should be high)")
                            print(f"     Negative pairs: {neg_sim:.4f} (should be low)")
                            if pos_sim <= neg_sim:
                                print("  ⚠️  WARNING: Positive pairs not more similar! Check pairing!")
                    
                    # SupConLoss 형식으로 reshape: [batch_size, 2, feature_dim]
                    features = torch.stack([f1, f2], dim=1)
                    
                    # 차원 확인
                    assert features.shape[0] == batch_labels.shape[0], \
                        f"Batch size mismatch: features {features.shape[0]} vs labels {batch_labels.shape[0]}"
                    assert features.shape[1] == 2, \
                        f"Must have 2 views, got {features.shape[1]}"
                    
                    # Calculate loss
                    loss = self.criterion(features, batch_labels)
                    loss_avg.update(loss.item(), batch_size)
                    
                    # Backward with gradient clipping
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / self.config.training.iterations_per_epoch
                print(f"  Epoch [{epoch+1}/{self.config.training.scr_epochs}] Loss: {avg_loss:.4f}")
        
        # 5. NCM 업데이트 (메모리 버퍼 데이터만 사용) 😶‍🌫️
        # self._update_ncm() 😶‍🌫️
        
        # 6. 메모리 버퍼 업데이트 😶‍🌫️
        # 🌈 올바른 순서: 버퍼 먼저, NCM 나중!
        print(f"\n=== Before buffer update ===")
        print(f"Buffer size: {len(self.memory_buffer)}")
        print(f"Buffer seen classes: {self.memory_buffer.seen_classes if hasattr(self.memory_buffer, 'seen_classes') else 'N/A'}")
        print(f"image_paths: {type(image_paths)}, len: {len(image_paths)}")
        print(f"original_labels: {type(original_labels)}, len: {len(original_labels)}")
        print(f"original_labels content: {original_labels}")
        
        # 🌈 Step 1: 메모리 버퍼 업데이트 (먼저!)
        self.memory_buffer.update_from_dataset(image_paths, original_labels)  # 원본 labels 사용
        self._full_memory_dataset = None  # 캐시 무효화
        print(f"Memory buffer size after update: {len(self.memory_buffer)}")
        
        # 🌈 Step 2: NCM 업데이트 (버퍼 업데이트 후!)
        self._update_ncm()
        
        # 🌈 디버깅: NCM과 버퍼 동기화 확인
        all_paths, all_labels = self.memory_buffer.get_all_data()
        buffer_classes = set(int(label) for label in all_labels)
        ncm_classes = set(self.ncm.class_means_dict.keys())
        missing = buffer_classes - ncm_classes
        
        if missing:
            print(f"⚠️  NCM missing classes: {sorted(list(missing))}")
        else:
            print(f"✅ NCM synchronized: {len(ncm_classes)} classes")
        
        # print(f"Memory buffer size: {len(self.memory_buffer)}") 😶‍🌫️
        
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
        # 🌈 momentum 수정: 완전 교체 방식으로 변경
        # self.ncm.update_class_means_dict( 😶‍🌫️
        #     class_means,  😶‍🌫️
        #     momentum=self.config.training.ncm_momentum 😶‍🌫️
        # ) 😶‍🌫️
        
        # 🌈 방법 1: 완전 교체 (추천)
        self.ncm.replace_class_means_dict(class_means)
        print(f"Updated NCM with {len(class_means)} classes (full replacement)")
        
        # 🌈 방법 2: 선택적 momentum (옵션 - 필요시 주석 해제)
        # for label, new_mean in class_means.items():
        #     if label not in self.ncm.class_means_dict:
        #         # 새 클래스: momentum 0
        #         self.ncm.update_class_means_dict({label: new_mean}, momentum=0.0)
        #     else:
        #         # 기존 클래스: config momentum 적용
        #         self.ncm.update_class_means_dict(
        #             {label: new_mean}, 
        #             momentum=self.config.training.ncm_momentum
        #         )
        
        # print(f"Updated NCM with {len(class_means)} classes") 😶‍🌫️
        
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