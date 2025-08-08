# utils/pretrained_loader.py (간소화)
import torch
import torch.nn as nn
from pathlib import Path

class PretrainedLoader:
    """CCNet 사전훈련 모델 로더 (ArcLayer 제거 버전)"""
    
    @staticmethod
    def load_ccnet_pretrained(
        model: nn.Module, 
        checkpoint_path: Path,
        device: str = 'cuda',
        verbose: bool = True
    ) -> nn.Module:
        """
        ArcLayer가 제거된 CCNet 사전훈련 가중치 로드
        """
        print("\n" + "="*60)
        print("🔥 Loading Pretrained CCNet (No ArcLayer)")
        print("="*60)
        print(f"📁 Path: {checkpoint_path}")
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # 1. 체크포인트 로드
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 2. state_dict 추출
        if isinstance(checkpoint, dict):
            # 여러 형태 지원
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # 3. 현재 모델의 state_dict
        model_dict = model.state_dict()
        
        # 4. 매칭되는 레이어만 로드
        loaded = {}
        skipped = []
        
        for name, param in state_dict.items():
            # DataParallel prefix 제거
            name = name.replace('module.', '')
            
            # ArcLayer 건너뛰기 (혹시 있다면)
            if 'arc' in name.lower() or 'classifier' in name:
                skipped.append(name)
                continue
            
            # 매칭되는 레이어만 로드
            if name in model_dict:
                if model_dict[name].shape == param.shape:
                    loaded[name] = param
                    if verbose:
                        print(f"  ✅ {name}")
                else:
                    skipped.append(f"{name} (shape mismatch)")
            else:
                skipped.append(f"{name} (not found)")
        
        # 5. 통계 출력
        print(f"\n📊 Loaded: {len(loaded)}/{len(state_dict)} layers")
        if skipped and verbose:
            print(f"⚠️  Skipped: {len(skipped)} layers")
        
        # 6. 가중치 업데이트
        model_dict.update(loaded)
        model.load_state_dict(model_dict, strict=False)
        
        print("✅ Pretrained weights loaded successfully!\n")
        print("="*60 + "\n")
        
        return model