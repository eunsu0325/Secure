#scr/scr_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, ConcatDataset
from torch.optim import lr_scheduler
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from tqdm import tqdm
from PIL import Image
import torch.nn.functional as F

from loss import SupConLoss, ProxyContrastLoss
from models import get_scr_transforms
from utils.util import AverageMeter
from utils.pretrained_loader import PretrainedLoader

# 오픈셋 관련 import
from scr.threshold_calculator import ThresholdCalibrator

# ⭐️ 에너지 버전 함수들 추가
try:
    from utils.utils_openset import (
        split_user_data,
        extract_scores_genuine,
        extract_scores_impostor_between,
        extract_scores_impostor_unknown,
        extract_scores_impostor_negref,
        balance_impostor_scores,
        predict_batch,
        load_paths_labels_from_txt,
        # ⭐️ 에너지 버전들 (있으면 import)
        extract_scores_genuine_energy,
        extract_scores_impostor_between_energy,
        extract_scores_impostor_negref_energy
    )
    ENERGY_FUNCTIONS_AVAILABLE = True
except ImportError:
    from utils.utils_openset import (
        split_user_data,
        extract_scores_genuine,
        extract_scores_impostor_between,
        extract_scores_impostor_unknown,
        extract_scores_impostor_negref,
        balance_impostor_scores,
        predict_batch,
        load_paths_labels_from_txt
    )
    ENERGY_FUNCTIONS_AVAILABLE = False
    print("⚠️ Energy score functions not available, using max score only")


def _ensure_single_view(data):
    """데이터가 2뷰 형식이면 첫 번째 뷰만 반환"""
    if isinstance(data, (list, tuple)) and len(data) == 2:
        return data[0]
    return data


class MemoryDataset(Dataset):
    def __init__(self, paths: List[str], labels: List[int], transform, train=True, dual_views=None):
        self.paths = paths
        
        if torch.is_tensor(labels):
            self.labels = labels.cpu().numpy()
        else:
            self.labels = np.array(labels)
            
        self.transform = transform
        self.train = train
        self.dual_views = dual_views if dual_views is not None else train
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        path = self.paths[index]
        label = self.labels[index]
        
        img = Image.open(path).convert('L')
        
        if self.dual_views:
            data1 = self.transform(img)
            data2 = self.transform(img)
            return [data1, data2], label
        else:
            data = self.transform(img)
            return data, label


class SCRTrainer:
    """
    Supervised Contrastive Replay Trainer with Open-set Support.
    💎 Enhanced with ProxyContrastLoss and Coverage Sampling
    ⭐️ Energy Score Support Added
    """
    
    def __init__(self,
                 model: nn.Module,
                 ncm_classifier,
                 memory_buffer,
                 config,
                 device='cuda'):
        
        # 사전훈련 로딩
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
        
        # 💎 ProxyContrastLoss 초기화
        self.use_proxy = getattr(config.training, 'use_proxy_loss', True)
        self.proxy_lambda = getattr(config.training, 'proxy_lambda', 0.3)
        self.proxy_temp = getattr(config.training, 'proxy_temperature', 0.15)
        self.proxy_topk = getattr(config.training, 'proxy_topk', 30)
        self.proxy_full_until = getattr(config.training, 'proxy_full_until', 100)
        self.proxy_min_classes = getattr(config.training, 'proxy_warmup_classes', 5)
        
        if self.use_proxy:
            self.proxy_loss = ProxyContrastLoss(
                temperature=self.proxy_temp,
                lambda_proxy=self.proxy_lambda,
                topk=self.proxy_topk,
                full_until=self.proxy_full_until
            )
            print(f"💎 ProxyContrastLoss enabled:")
            print(f"   Temperature: {self.proxy_temp} (vs SupCon: {config.training.temperature})")
            print(f"   Lambda: {self.proxy_lambda}")
            print(f"   Top-K: {self.proxy_topk}")
        else:
            self.proxy_loss = None
        
        # 💎 프로토타입 캐시
        self.proto_cache_P = None
        self.proto_cache_ids = []
        self.proxy_active = False
        
        # 💎 커버리지 샘플링 설정
        self.use_coverage = getattr(config.training, 'use_coverage_sampling', True)
        self.coverage_k = getattr(config.training, 'coverage_k_per_class', 2)
        
        if self.use_coverage:
            print(f"💎 Coverage sampling enabled: k={self.coverage_k} per class")
        
        # 💎 Lambda 스케줄 설정
        self.proxy_lambda_schedule = getattr(config.training, 'proxy_lambda_schedule', None)
        if self.proxy_lambda_schedule is None and self.use_proxy:
            self.proxy_lambda_schedule = {
                'warmup': 0.1,
                'warmup_10': 0.2,
                'warmup_20': 0.3
            }
            print(f"💎 Lambda schedule: {self.proxy_lambda_schedule}")
        
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
        
        # 메모리 데이터셋 캐시
        self._full_memory_dataset = None
        self._memory_size_at_last_update = 0
        
        # === 오픈셋 관련 초기화 ===
        self.openset_enabled = hasattr(config, 'openset') and config.openset.enabled
        
        if self.openset_enabled:
            self.openset_config = config.openset
            self._first_calibration_done = False  # ← 추가
            
            # ⭐️ 에너지 스코어 설정 확인
            self.use_energy_score = getattr(config.openset, 'score_mode', 'max') == 'energy'
            if self.use_energy_score and ENERGY_FUNCTIONS_AVAILABLE:
                # NCM에 에너지 설정 적용
                self.ncm.set_energy_config(
                    use_energy=True,
                    T=getattr(config.training, 'energy_temperature', 0.15),
                    k_mode=getattr(config.training, 'energy_k_mode', 'sqrt'),
                    k_fixed=getattr(config.training, 'energy_k_fixed', 10)
                )
                print(f"⭐️ Energy Score Mode Enabled:")
                print(f"   Temperature: {self.ncm.energy_T}")
                print(f"   K mode: {self.ncm.energy_k_mode}")
                if self.ncm.energy_k_mode == 'fixed':
                    print(f"   K fixed: {self.ncm.energy_k_fixed}")
            elif self.use_energy_score and not ENERGY_FUNCTIONS_AVAILABLE:
                print("⚠️ Energy score requested but functions not available, using max score")
                self.use_energy_score = False
            
            # ThresholdCalibrator 초기화 (⚡️ 마진 제거)
            self.threshold_calibrator = ThresholdCalibrator(
                mode="cosine",
                threshold_mode=config.openset.threshold_mode,
                target_far=config.openset.target_far,
                alpha=config.openset.threshold_alpha,
                max_delta=config.openset.threshold_max_delta,
                clip_range=(-1.0, 1.0),
                # ⚡️ use_auto_margin 제거
                # ⚡️ margin_init 제거
                far_target=None,
                min_samples=10,
                verbose=config.openset.verbose_calibration
            )
            
            # Dev/Train 데이터 관리
            self.dev_data = {}
            self.train_data = {}
            self.registered_users = set()
            
            # 평가 히스토리
            self.evaluation_history = []
            
            # 초기 임계치 설정 (⚡️ 마진 제거)
            self.ncm.set_thresholds(
                tau_s=config.openset.initial_tau
                # ⚡️ use_margin, tau_m 제거
            )
            
            # 모드별 초기값 조정
            if config.openset.threshold_mode == 'far':
                print("🎯 FAR Target mode enabled")
                print(f"   Target FAR: {config.openset.target_far*100:.1f}%")
            else:
                print("🐋 EER mode enabled")
            
            print(f"   Initial τ_s: {config.openset.initial_tau}")
            # ⚡️ 마진 출력 제거
        else:
            self.registered_users = set()
            print("📌 Open-set mode disabled")
        
        # Statistics
        self.experience_count = 0
        
        # BASE_ID 저장
        self.base_id = int(config.negative.base_id) if hasattr(config, 'negative') else 10000
        print(f"🌽 SCRTrainer using BASE_ID: {self.base_id}")
        
        # 워밍업 관련 파라미터
        self.warmup_T = int(config.negative.warmup_experiences) if hasattr(config, 'negative') else 4
        self.r0 = float(config.negative.r0) if hasattr(config, 'negative') else 0.5
        self.max_neg_per_class = int(config.negative.max_per_batch) if hasattr(config, 'negative') else 1
        
        # CuDNN 최적화
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            print("🚀 CuDNN benchmark enabled for fixed-size inputs")
        
        # 사전훈련 사용 여부 로그
        if hasattr(config.model, 'use_pretrained') and config.model.use_pretrained:
            print(f"🔥 SCRTrainer initialized with pretrained model")
        else:
            print(f"🎲 SCRTrainer initialized with random weights")
        
        # 💎 모니터링 변수
        self.proxy_loss_history = []
        self.drift_history = []

    def _dedup_negative_classes(self, paths, labels, max_per_class=1):
        """네거티브 클래스 중복 제거"""
        out_p, out_l, seen = [], [], set()
        for p, l in zip(paths, labels):
            li = int(l)
            if li >= self.base_id:
                if (li in seen) and (max_per_class == 1):
                    continue
                seen.add(li)
            out_p.append(p)
            out_l.append(l)
        return out_p, out_l
    
    def _memory_cap_for_ratio(self, num_current, r):
        """r0 비율 제한 계산"""
        if r <= 0:
            return 0
        return int((r / max(1e-8, 1.0 - r)) * num_current)
    
    def get_proxy_lambda(self):
        """Experience 기반 lambda 스케줄링"""
        if not self.proxy_lambda_schedule or not self.use_proxy:
            return self.proxy_lambda
        
        lambda_val = 0.0
        
        if self.experience_count >= self.warmup_T:
            lambda_val = self.proxy_lambda_schedule.get('warmup', 0.1)
        if self.experience_count >= self.warmup_T + 10:
            lambda_val = self.proxy_lambda_schedule.get('warmup_10', 0.2)
        if self.experience_count >= self.warmup_T + 20:
            lambda_val = self.proxy_lambda_schedule.get('warmup_20', 0.3)
        
        return lambda_val
    
    @torch.no_grad()
    def update_proto_cache(self):
        """💎 6144D NCM 프로토타입을 128D로 투영하여 캐시"""
        if self.experience_count < self.warmup_T:
            return
        
        real_classes = [c for c in self.ncm.class_means_dict.keys()
                        if c < self.base_id]
        
        if len(real_classes) < self.proxy_min_classes:
            print(f"⏳ Waiting for more classes: {len(real_classes)}/{self.proxy_min_classes}")
            return
        
        self.proxy_active = True
        self.proto_cache_ids = sorted(real_classes)
        
        protos_6144 = torch.stack([
            self.ncm.class_means_dict[c]
            for c in self.proto_cache_ids
        ]).to(self.device)
        
        self.model.eval()
        protos_6144_norm = F.normalize(protos_6144, dim=-1)
        P = self.model.projection_head(protos_6144_norm)
        self.proto_cache_P = F.normalize(P, dim=-1).detach()
        self.model.train()
        
        print(f"💎 Proto cache updated: {len(self.proto_cache_ids)} classes")
        print(f"   Cache shape: {self.proto_cache_P.shape}")
    
    def train_experience(self, user_id: int, image_paths: List[str], labels: List[int]) -> Dict:
        """하나의 experience (한 명의 사용자) 학습 - 오픈셋 지원."""
        
        print(f"\n=== Training Experience {self.experience_count}: User {user_id} ===")
        
        # 💎 Lambda 스케줄 업데이트
        if self.use_proxy and self.proxy_loss:
            new_lambda = self.get_proxy_lambda()
            self.proxy_loss.lambda_proxy = new_lambda
            if self.experience_count % 10 == 0:
                print(f"💎 Lambda schedule: {new_lambda:.2f}")
        
        # 💎 메모리 버퍼에 경험 카운터 전달 (로그용)
        if hasattr(self.memory_buffer, 'set_experience_count'):
            self.memory_buffer.set_experience_count(self.experience_count)
        
        # 원본 labels를 보존 (중요!)
        original_labels = labels.copy()
        
        # Step 1: Dev/Train 분리 (오픈셋 모드인 경우)
        if self.openset_enabled:
            train_paths, train_labels, dev_paths, dev_labels = split_user_data(
                image_paths, original_labels,
                dev_ratio=self.openset_config.dev_ratio,
                min_dev=2
            )
            
            # 저장
            self.dev_data[user_id] = (dev_paths, dev_labels)
            self.train_data[user_id] = (train_paths, train_labels)
            
            print(f"🐋 Data split: Train={len(train_paths)}, Dev={len(dev_paths)}")
        else:
            train_paths = image_paths
            train_labels = original_labels
        
        self.registered_users.add(user_id)
        
        # 현재 사용자 데이터셋 생성 (Train 데이터만)
        current_dataset = MemoryDataset(
            paths=train_paths,
            labels=train_labels,
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
                
                # 1. 현재 데이터에서 B_n 샘플링
                current_indices = np.random.choice(
                    len(current_dataset),
                    size=min(len(current_dataset), self.config.training.scr_batch_size),
                    replace=False
                )
                current_subset = Subset(current_dataset, current_indices)
                
                # 2. 메모리에서 B_M 샘플링
                if len(self.memory_buffer) > 0:
                    # 💎 커버리지 샘플링 사용
                    if self.use_coverage and hasattr(self.memory_buffer, 'coverage_sample'):
                        memory_paths, memory_labels = self.memory_buffer.coverage_sample(
                            self.config.training.memory_batch_size,
                            k_per_class=self.coverage_k
                        )
                        if iteration == 0 and epoch == 0:
                            print(f"💎 Using coverage sampling")
                    else:
                        memory_paths, memory_labels = self.memory_buffer.sample(
                            self.config.training.memory_batch_size
                        )
                    
                    if torch.is_tensor(memory_labels):
                        memory_labels = memory_labels.cpu().tolist()
                    
                    # 워밍업 규칙 적용
                    if self.experience_count < self.warmup_T:
                        memory_paths, memory_labels = self._dedup_negative_classes(
                            memory_paths, memory_labels, max_per_class=self.max_neg_per_class
                        )
                        
                        cap = self._memory_cap_for_ratio(len(current_indices), self.r0)
                        if cap >= 0 and len(memory_paths) > cap:
                            idx = np.random.choice(len(memory_paths), size=cap, replace=False)
                            memory_paths = [memory_paths[i] for i in idx]
                            memory_labels = [memory_labels[i] for i in idx]
                    else:
                        kept = [(p, l) for p, l in zip(memory_paths, memory_labels) if int(l) < self.base_id]
                        memory_paths, memory_labels = (list(map(list, zip(*kept))) if kept else ([], []))
                    
                    neg_in_mem = sum(1 for l in memory_labels if int(l) >= self.base_id)
                    if iteration == 0 and epoch == 0:
                        print(f"[DEBUG][exp{self.experience_count}] cur={len(current_indices)}, mem={len(memory_paths)}, neg_in_mem={neg_in_mem}, r0={self.r0}")
                    
                    if memory_paths:
                        memory_dataset = MemoryDataset(
                            paths=memory_paths,
                            labels=memory_labels,
                            transform=self.train_transform,
                            train=True
                        )
                        
                        combined_dataset = ConcatDataset([current_subset, memory_dataset])
                    else:
                        combined_dataset = current_subset
                        if self.experience_count == self.warmup_T:
                            print("📌 First post-warmup experience: memory empty, using current only")
                else:
                    combined_dataset = current_subset
                
                # 4. DataLoader로 배치 생성
                batch_loader = DataLoader(
                    combined_dataset,
                    batch_size=len(combined_dataset),
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    persistent_workers=False
                )
                
                # 5. 학습
                for data, batch_labels in batch_loader:
                    batch_size = len(batch_labels)
                    
                    view1 = data[0]
                    view2 = data[1]
                    
                    x = torch.cat([view1, view2], dim=0).to(self.device, non_blocking=True)
                    batch_labels = batch_labels.to(self.device, non_blocking=True)
                    
                    self.optimizer.zero_grad()
                    features_all = self.model(x)
                    
                    f1 = features_all[:batch_size]
                    f2 = features_all[batch_size:]
                    
                    if iteration == 0 and epoch == 0:
                        neg_in_batch = int((batch_labels >= self.base_id).sum().item())
                        real_in_batch = int(len(batch_labels) - neg_in_batch)
                        print(f"[DEBUG][exp{self.experience_count}] batch real={real_in_batch}, neg={neg_in_batch}")
                        
                        with torch.no_grad():
                            pos_sim = torch.nn.functional.cosine_similarity(f1, f2, dim=1).mean()
                            neg_sim = torch.nn.functional.cosine_similarity(
                                f1, f2.roll(shifts=1, dims=0), dim=1
                            ).mean()
                            print(f"  📊 Similarity check:")
                            print(f"     Positive pairs: {pos_sim:.4f} (should be high)")
                            print(f"     Negative pairs: {neg_sim:.4f} (should be low)")
                            if pos_sim <= neg_sim:
                                print("  ⚠️  WARNING: Positive pairs not more similar! Check pairing!")
                    
                    features_paired = torch.stack([f1, f2], dim=1)
                    
                    assert features_paired.shape[0] == batch_labels.shape[0], \
                        f"Batch size mismatch: features {features_paired.shape[0]} vs labels {batch_labels.shape[0]}"
                    assert features_paired.shape[1] == 2, \
                        f"Must have 2 views, got {features_paired.shape[1]}"
                    
                    loss = self.criterion(features_paired, batch_labels)
                    
                    # 💎 ProxyContrastLoss 추가
                    if self.proxy_active and self.proto_cache_P is not None and self.use_proxy:
                        z_all = F.normalize(features_all, dim=-1)
                        y_all = batch_labels.repeat(2).long()
                        
                        loss_proxy = self.proxy_loss(
                            z_all, y_all,
                            self.proto_cache_P,
                            self.proto_cache_ids
                        )
                        
                        loss = loss + loss_proxy
                        
                        if iteration == 0 and epoch == 0:
                            print(f"💎 ProxyLoss: {loss_proxy.item():.4f}, SupCon: {loss.item() - loss_proxy.item():.4f}")
                            self.proxy_loss_history.append(loss_proxy.item())
                    
                    loss_avg.update(loss.item(), batch_size)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
            
            if (epoch + 1) % 5 == 0:
                avg_loss = epoch_loss / self.config.training.iterations_per_epoch
                print(f"  Epoch [{epoch+1}/{self.config.training.scr_epochs}] Loss: {avg_loss:.4f}")

        # 메모리 버퍼 업데이트
        print(f"\n=== Before buffer update ===")
        print(f"Buffer size: {len(self.memory_buffer)}")
        print(f"Buffer seen classes: {self.memory_buffer.seen_classes if hasattr(self.memory_buffer, 'seen_classes') else 'N/A'}")
        
        self.memory_buffer.update_from_dataset(train_paths, train_labels)
        self._full_memory_dataset = None
        print(f"Memory buffer size after update: {len(self.memory_buffer)}")
        
        # NCM 업데이트
        self._update_ncm()
        
        # 💎 프로토타입 캐시 업데이트
        if self.use_proxy:
            self.update_proto_cache()
        
        # 디버깅: NCM과 버퍼 동기화 확인
        all_paths, all_labels = self.memory_buffer.get_all_data()
        buffer_classes = set(int(label) for label in all_labels)
        ncm_classes = set(self.ncm.class_means_dict.keys())
        missing = buffer_classes - ncm_classes
        
        if missing:
            print(f"⚠️  NCM missing classes: {sorted(list(missing))}")
        else:
            print(f"✅ NCM synchronized: {len(ncm_classes)} classes")
        
        # 주기적 캘리브레이션 및 평가 (오픈셋 모드)
        if self.openset_enabled and len(self.registered_users) % self.config.training.test_interval == 0:
            
            if len(self.registered_users) >= self.openset_config.warmup_users:
                print("\n" + "="*60)
                print("🔄 THRESHOLD CALIBRATION & EVALUATION")
                print("="*60)
                
                self._calibrate_threshold()
                metrics = self._evaluate_openset()
                
                self.evaluation_history.append({
                    'num_users': len(self.registered_users),
                    'tau_s': self.ncm.tau_s,
                    # ⚡️ tau_m 제거
                    'metrics': metrics
                })
                
                print("="*60 + "\n")
            else:
                print(f"📌 Warmup phase: {len(self.registered_users)}/{self.openset_config.warmup_users}")
        
        self.experience_count += 1
        self.scheduler.step()
        
        return {
            'experience': self.experience_count - 1,
            'user_id': user_id,
            'loss': loss_avg.avg,
            'memory_size': len(self.memory_buffer),
            'num_registered': len(self.registered_users),
            'neg_in_batch': neg_in_mem if 'neg_in_mem' in locals() else 0
        }
    
    @torch.no_grad()
    def _calibrate_threshold(self):
        """임계치 캘리브레이션 (⭐️ 에너지 모드 지원)"""
        
        # ⭐️ 스코어 모드 확인
        score_mode = getattr(self.openset_config, 'score_mode', 'max')
        
        # ⭐️ 모드별 설정
        if score_mode == 'energy' and self.use_energy_score and ENERGY_FUNCTIONS_AVAILABLE:
            self.ncm.use_energy = True
            print(f"⭐️ Using ENERGY scores for calibration")
            print(f"   T={self.ncm.energy_T}, k_mode={self.ncm.energy_k_mode}")
        else:
            self.ncm.use_energy = False
            print(f"📊 Using MAX scores for calibration")
        
        print(f"\n📊 Extracting scores for {self.openset_config.threshold_mode.upper()} calibration...")
        
        # Dev 데이터 수집
        all_dev_paths = []
        all_dev_labels = []
        for uid, (paths, labels) in self.dev_data.items():
            all_dev_paths.extend(paths)
            all_dev_labels.extend([uid] * len(paths))
        
        # ⭐️ 모드에 따른 점수 추출
        if score_mode == 'energy' and self.use_energy_score and ENERGY_FUNCTIONS_AVAILABLE:
            # 에너지 버전 사용
            s_genuine = extract_scores_genuine_energy(
                self.model, self.ncm,
                all_dev_paths, all_dev_labels,
                self.test_transform, self.device
            )
            
            s_imp_between = extract_scores_impostor_between_energy(
                self.model, self.ncm,
                all_dev_paths, all_dev_labels,
                self.test_transform, self.device,
                max_pairs=2000
            )
            
            s_imp_negref = extract_scores_impostor_negref_energy(
                self.model, self.ncm,
                self.config.dataset.negative_samples_file,
                self.test_transform, self.device,
                max_eval=self.openset_config.negref_max_eval
            )
        else:
            # 기존 최댓값 버전
            s_genuine = extract_scores_genuine(
                self.model, self.ncm,
                all_dev_paths, all_dev_labels,
                self.test_transform, self.device
            )
            
            s_imp_between = extract_scores_impostor_between(
                self.model, self.ncm,
                all_dev_paths, all_dev_labels,
                self.test_transform, self.device,
                max_pairs=2000
            )
            
            s_imp_negref = extract_scores_impostor_negref(
                self.model, self.ncm,
                self.config.dataset.negative_samples_file,
                self.test_transform, self.device,
                max_eval=self.openset_config.negref_max_eval
            )
        
        # 균형 맞추기
        s_impostor = balance_impostor_scores(
            s_imp_between,
            None,  # Unknown 제거
            s_imp_negref,
            ratio=(0.3, 0.0, 0.7)
        )
        
        print(f"   Genuine: {len(s_genuine)} scores")
        print(f"   Impostor: {len(s_impostor)} scores (Between + NegRef)")
        
        # 캘리브레이션
        if len(s_genuine) >= 10 and len(s_impostor) >= 10:
            result = self.threshold_calibrator.calibrate(
                genuine_scores=s_genuine,
                impostor_scores=s_impostor,
                old_tau=None if not self._first_calibration_done else self.ncm.tau_s  # ← 수정
            )
            self._first_calibration_done = True  # ← 추가
            # NCM에 적용 (⚡️ 마진 제거)
            self.ncm.set_thresholds(
                tau_s=result['tau_smoothed']
                # ⚡️ use_margin, tau_m 제거
            )
            
            # 모드별 출력
            if self.openset_config.threshold_mode == 'eer':
                print(f"📊 EER Mode Results:")
                print(f"   EER: {result.get('eer', 0)*100:.2f}%")
                print(f"   Threshold: {result['tau_smoothed']:.4f}")
            else:
                print(f"🎯 FAR Target Results:")
                print(f"   Target FAR: {self.openset_config.target_far*100:.1f}%")
                print(f"   Achieved FAR: {result.get('current_far', 0)*100:.1f}%")
                print(f"   Threshold: {result['tau_smoothed']:.4f}")
            
            # ⭐️ 에너지 모드면 추가 정보
            if score_mode == 'energy' and self.use_energy_score:
                print(f"   ⭐️ Energy params: T={self.ncm.energy_T}, k_mode={self.ncm.energy_k_mode}")
        else:
            print("⚠️ Not enough samples for calibration")
    
    @torch.no_grad()
    def _evaluate_openset(self):
        """오픈셋 평가"""
        
        print("\n📈 Open-set Evaluation:")
        
        # 1. Known-Dev (TAR/FRR)
        all_dev_paths = []
        all_dev_labels = []
        for uid, (paths, labels) in self.dev_data.items():
            all_dev_paths.extend(paths)
            all_dev_labels.extend([uid] * len(paths))
        
        if all_dev_paths:
            preds = predict_batch(
                self.model, self.ncm,
                all_dev_paths, self.test_transform, self.device
            )
            
            correct = sum(1 for p, l in zip(preds, all_dev_labels) if p == l)
            rejected = sum(1 for p in preds if p == -1)
            
            TAR = correct / max(1, len(preds))
            FRR = rejected / max(1, len(preds))
        else:
            TAR = FRR = 0.0
        
        # 2. Unknown (TRR/FAR)
        unknown_paths, unknown_labels = load_paths_labels_from_txt(
            self.config.dataset.train_set_file
        )
        
        unknown_filtered = []
        for p, l in zip(unknown_paths, unknown_labels):
            if l not in self.registered_users:
                unknown_filtered.append(p)
        
        if len(unknown_filtered) > 1000:
            unknown_filtered = np.random.choice(unknown_filtered, 1000, replace=False).tolist()
        
        if unknown_filtered:
            preds_unk = predict_batch(
                self.model, self.ncm,
                unknown_filtered, self.test_transform, self.device
            )
            TRR_u = sum(1 for p in preds_unk if p == -1) / len(preds_unk)
            FAR_u = 1 - TRR_u
        else:
            TRR_u = FAR_u = 0.0
        
        # 3. NegRef (선택)
        negref_paths, _ = load_paths_labels_from_txt(
            self.config.dataset.negative_samples_file
        )
        
        if len(negref_paths) > 1000:
            negref_paths = np.random.choice(negref_paths, 1000, replace=False).tolist()
        
        if negref_paths:
            preds_neg = predict_batch(
                self.model, self.ncm,
                negref_paths, self.test_transform, self.device
            )
            TRR_n = sum(1 for p in preds_neg if p == -1) / len(preds_neg)
            FAR_n = 1 - TRR_n
        else:
            TRR_n = FAR_n = None
        
        # 결과 출력
        print(f"   Known-Dev: TAR={TAR:.3f}, FRR={FRR:.3f}")
        print(f"   Unknown: TRR={TRR_u:.3f}, FAR={FAR_u:.3f}")
        if TRR_n is not None:
            print(f"   NegRef: TRR={TRR_n:.3f}, FAR={FAR_n:.3f}")
        print(f"   Threshold: τ_s={self.ncm.tau_s:.4f}")
        
        # 모드 정보 추가
        if self.openset_config.threshold_mode == 'far':
            print(f"   Mode: FAR Target ({self.openset_config.target_far*100:.1f}%)")
        else:
            print(f"   Mode: EER")
        
        # ⭐️ 에너지 모드 표시
        if hasattr(self, 'use_energy_score') and self.use_energy_score:
            print(f"   ⭐️ Score: Energy (T={self.ncm.energy_T})")
        else:
            print(f"   Score: Max")
        
        return {
            'TAR': TAR, 'FRR': FRR,
            'TRR_unknown': TRR_u, 'FAR_unknown': FAR_u,
            'TRR_negref': TRR_n, 'FAR_negref': FAR_n,
            'mode': self.openset_config.threshold_mode,
            'score_type': 'energy' if self.use_energy_score else 'max'  # ⭐️
        }
    
    @torch.no_grad()
    def _update_ncm(self):
        """NCM classifier의 class means를 업데이트합니다."""
        if len(self.memory_buffer) == 0:
            return
            
        self.model.eval()
        
        # 메모리 버퍼에서 모든 데이터 가져오기
        all_paths, all_labels = self.memory_buffer.get_all_data()
        
        # 가짜 클래스 필터링
        real_paths = []
        real_labels = []
        fake_count = 0
        
        for path, label in zip(all_paths, all_labels):
            label_int = int(label) if not isinstance(label, int) else label
            if label_int < self.base_id:
                real_paths.append(path)
                real_labels.append(label)
            else:
                fake_count += 1
        
        if not real_paths:
            print("⚠️ No real users for NCM update (NCM remains empty)")
            return
        
        if fake_count > 0:
            print(f"🍄 NCM update: {len(real_paths)} real samples ({fake_count} fake samples filtered out)")
        
        dataset = MemoryDataset(
            paths=real_paths,
            labels=real_labels,
            transform=self.test_transform,
            train=False
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            persistent_workers=False
        )
        
        # 클래스별로 features 수집
        class_features = {}
        
        for data, labels in dataloader:
            data = _ensure_single_view(data).to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            features = self.model.getFeatureCode(data)
            
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
                mean_feature = mean_feature / (mean_feature.norm(p=2) + 1e-12)
                class_means[label] = mean_feature
        
        # NCM 업데이트
        self.ncm.replace_class_means_dict(class_means)
        
        real_classes = [k for k in class_means.keys() if k < self.base_id]
        print(f"🍄 Updated NCM with {len(real_classes)} real classes (no fake contamination)")
        
        self.model.train()
    
    def evaluate(self, test_dataset: Dataset) -> float:
        """NCM을 사용하여 정확도를 평가합니다."""
        self.model.eval()
        
        dataloader = DataLoader(
            test_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True,
            persistent_workers=False
        )
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in dataloader:
                data = _ensure_single_view(data).to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                features = self.model.getFeatureCode(data)
                predictions = self.ncm.predict(features)
                
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        
        accuracy = 100.0 * correct / total
        return accuracy
    
    def save_checkpoint(self, path: str):
        """체크포인트 저장"""
        checkpoint_dict = {
            'model_state_dict': self.model.state_dict(),
            'ncm_state_dict': self.ncm.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'experience_count': self.experience_count,
            'memory_buffer_size': len(self.memory_buffer)
        }
        
        # 💎 ProxyLoss 관련 저장
        if self.use_proxy:
            checkpoint_dict['proxy_data'] = {
                'proto_cache_P': self.proto_cache_P,
                'proto_cache_ids': self.proto_cache_ids,
                'proxy_active': self.proxy_active,
                'proxy_loss_history': self.proxy_loss_history
            }
        
        # 오픈셋 관련 추가 저장 (⚡️ tau_m 제거)
        if self.openset_enabled:
            checkpoint_dict['openset_data'] = {
                'tau_s': self.ncm.tau_s,
                # ⚡️ tau_m 제거
                'registered_users': list(self.registered_users),
                'evaluation_history': self.evaluation_history
            }
        
        torch.save(checkpoint_dict, path)
    
    def load_checkpoint(self, path: str):
        """체크포인트 로드"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.ncm.load_state_dict(checkpoint['ncm_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.experience_count = checkpoint['experience_count']
        
        # 💎 ProxyLoss 관련 복원
        if 'proxy_data' in checkpoint and self.use_proxy:
            proxy_data = checkpoint['proxy_data']
            self.proto_cache_P = proxy_data.get('proto_cache_P')
            self.proto_cache_ids = proxy_data.get('proto_cache_ids', [])
            self.proxy_active = proxy_data.get('proxy_active', False)
            self.proxy_loss_history = proxy_data.get('proxy_loss_history', [])
        
        # 오픈셋 관련 복원 (⚡️ tau_m 제거)
        if 'openset_data' in checkpoint and self.openset_enabled:
            openset_data = checkpoint['openset_data']
            self.ncm.tau_s = openset_data.get('tau_s')
            # ⚡️ self.ncm.tau_m 제거
            self.registered_users = set(openset_data.get('registered_users', []))
            self.evaluation_history = openset_data.get('evaluation_history', [])