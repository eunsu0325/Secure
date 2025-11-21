"""
🍑 테스트 스크립트: 6144D -> 512D projection 확인
"""
import torch
import sys
sys.path.append('/Users/kimeunsu/Downloads/secure/Secure')

from coconut.models.ccnet import ccnet

def test_projection_head():
    print("\n🍑 Testing Full Projection Head (6144D -> 512D)")
    print("="*60)

    # 모델 초기화
    model = ccnet(weight=0.8, use_projection=True, projection_dim=512)
    model.eval()

    # 테스트 입력
    batch_size = 4
    input_tensor = torch.randn(batch_size, 1, 128, 128)

    print(f"\n📥 Input shape: {input_tensor.shape}")

    # 1. Training mode forward (512D projection)
    model.train()
    with torch.no_grad():
        output_train = model(input_tensor)
    print(f"\n🏋️ Training mode output shape: {output_train.shape}")
    print(f"   Expected: [{batch_size}, 512]")
    assert output_train.shape == (batch_size, 512), f"Training output shape mismatch!"

    # 2. Eval mode forward (6144D)
    model.eval()
    with torch.no_grad():
        output_eval = model(input_tensor)
    print(f"\n📊 Eval mode output shape: {output_eval.shape}")
    print(f"   Expected: [{batch_size}, 6144]")
    assert output_eval.shape == (batch_size, 6144), f"Eval output shape mismatch!"

    # 3. getFeatureCode without projection (6144D)
    with torch.no_grad():
        features_6144 = model.getFeatureCode(input_tensor, use_projection=False)
    print(f"\n🔢 getFeatureCode(projection=False) shape: {features_6144.shape}")
    print(f"   Expected: [{batch_size}, 6144]")
    assert features_6144.shape == (batch_size, 6144), f"6144D features shape mismatch!"

    # 4. getFeatureCode with projection (512D)
    with torch.no_grad():
        features_512 = model.getFeatureCode(input_tensor, use_projection=True)
    print(f"\n🎯 getFeatureCode(projection=True) shape: {features_512.shape}")
    print(f"   Expected: [{batch_size}, 512]")
    assert features_512.shape == (batch_size, 512), f"512D features shape mismatch!"

    print("\n✅ All tests passed! 🍑")
    print("="*60)

    # 성능 비교
    print("\n📈 Feature statistics:")
    print(f"   6144D mean: {features_6144.mean():.4f}, std: {features_6144.std():.4f}")
    print(f"   512D mean: {features_512.mean():.4f}, std: {features_512.std():.4f}")

    # 정규화 확인
    print("\n🔍 Normalization check:")
    norms_6144 = torch.norm(features_6144, dim=1)
    norms_512 = torch.norm(features_512, dim=1)
    print(f"   6144D L2 norms: {norms_6144}")
    print(f"   512D L2 norms: {norms_512}")

    if torch.allclose(norms_512, torch.ones_like(norms_512), atol=1e-5):
        print("   ✓ 512D features are L2-normalized")

    return True

if __name__ == "__main__":
    try:
        test_projection_head()
        print("\n🎉 Full Projection Head implementation successful!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()