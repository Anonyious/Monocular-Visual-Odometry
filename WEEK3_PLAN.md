# Week 3-4 Execution Plan: ScaleNet Training
## Path C2 — Learned Metric Scale Recovery

**Start Date**: 2026-09-12  
**Target Deliverable**: Trained `models/scale_net_v1.pth`  
**Status**: 🚀 In Progress

---

## 📊 Current Status

### ✅ Completed: Baseline Benchmarks (Week 1-2)

| Sequence | Frames | ATE RMSE | RPE RMSE | Status |
|----------|--------|----------|----------|--------|
| 01 | 1,101 | 958.36 m | 59.22 m | ✅ Complete |
| 02 | 4,661 | 422.11 m | 270.59 m | ✅ Complete |
| 03 | 801 | 89.24 m | 25.48 m | ✅ Complete |
| 05 | 2,761 | 98.22 m | 19.62 m | ✅ Complete |
| 06 | 1,101 | 109.26 m | 16.63 m | ✅ Complete |
| 08 | 4,071 | 73.71 m | 2.09 m | ✅ Complete |

**Total Frames**: 14,496  
**Baseline Established**: ✅ Yes

---

## 🎯 Week 3-4 Goals

### Phase 3.1: Data Preparation (Day 1-2)
- [ ] Create KITTI scale dataset loader
- [ ] Extract frame pairs with ground truth scales
- [ ] Implement data augmentation pipeline
- [ ] Train/val/test split (80/10/10)
- [ ] **Deliverable**: `scripts/prepare_scale_dataset.py`

### Phase 3.2: Model Training (Day 3-5)
- [ ] Verify ScaleNet architecture (`vo/scale_net.py`)
- [ ] Implement training loop with loss function
- [ ] Add validation monitoring
- [ ] Train for 10-15 epochs (~2-3 hours)
- [ ] **Deliverable**: `scripts/train_scale_net.py`

### Phase 3.3: Model Evaluation (Day 6-7)
- [ ] Evaluate on test set
- [ ] Generate loss curves
- [ ] Save best model checkpoint
- [ ] Quick integration test
- [ ] **Deliverable**: `models/scale_net_v1.pth` + training logs

---

## 📋 Implementation Steps

### Step 1: Create Dataset Loader

```python
# scripts/prepare_scale_dataset.py

"""
Extract (frame_i, frame_i+1, ground_truth_scale) triplets from KITTI.

Ground truth scale computation:
    scale = ||t_i+1 - t_i||  (translation magnitude between consecutive poses)
"""

import numpy as np
from pathlib import Path
from vo.data.kitti_loader import KITTISequence

def extract_scale_labels(sequence_dir, poses_file):
    """
    Args:
        sequence_dir: Path to sequence/XX/image_0/
        poses_file: Path to poses/XX.txt (ground truth)
    
    Returns:
        List of (frame_i_path, frame_i+1_path, scale) tuples
    """
    # Load ground truth poses
    poses = load_kitti_poses(poses_file)
    
    dataset = []
    for i in range(len(poses) - 1):
        T_i = poses[i]
        T_i_next = poses[i + 1]
        
        # Compute translation magnitude
        t_i = T_i[:3, 3]
        t_next = T_i_next[:3, 3]
        scale = np.linalg.norm(t_next - t_i)
        
        # Skip near-zero motion (camera stationary)
        if scale < 0.01:
            continue
        
        frame_i = sequence_dir / f"{i:06d}.png"
        frame_next = sequence_dir / f"{i+1:06d}.png"
        
        dataset.append((str(frame_i), str(frame_next), scale))
    
    return dataset
```

### Step 2: Training Script

```python
# scripts/train_scale_net.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from vo.scale_net import ScaleNet

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    
    for frames, scales_gt in loader:
        frames = frames.to(device)
        scales_gt = scales_gt.to(device)
        
        # Forward
        scale_pred, log_var = model(frames)
        
        # Loss: negative log-likelihood
        precision = torch.exp(-log_var)
        loss = torch.mean(
            precision * (scale_pred - scales_gt)**2 + log_var
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

# Training loop
model = ScaleNet().to('cuda')
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(15):
    train_loss = train_epoch(model, train_loader, optimizer, 'cuda')
    val_loss = validate(model, val_loader, 'cuda')
    
    print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Save best model
    if val_loss < best_val_loss:
        torch.save(model.state_dict(), 'models/scale_net_v1.pth')
        best_val_loss = val_loss
```

### Step 3: Data Augmentation

- **Geometric**: Random crop, resize, horizontal flip
- **Photometric**: Brightness/contrast jitter, Gaussian blur
- **Synthetic scale variation**: Random zoom (0.8x - 1.2x)

---

## 💾 Expected Outputs

```
models/
├── scale_net_v1.pth           # Best model checkpoint
├── training_log.txt           # Loss curves
└── scale_net_config.json      # Hyperparameters

results/
└── scale_net_training/
    ├── train_loss.png
    ├── val_loss.png
    └── test_metrics.json
```

---

## 📈 Success Criteria

- ✅ Training converges (loss decreases steadily)
- ✅ Validation loss < 0.5 (scale error < 0.5m typically)
- ✅ Test set scale RMSE < 0.3m
- ✅ No overfitting (val loss tracks train loss)
- ✅ Model file saved successfully

---

## 🚀 Next Steps After Week 3-4

**Week 5-6**: Integrate ScaleNet into `vo/odometry.py`
- Replace `GroundPlaneScaleRecovery` with `ScaleRecoveryNetwork`
- Add `--use_learned_scale` flag to benchmark scripts
- Run comparison: RANSAC vs. Learned

---

## 📞 Dependencies

**Required**:
- ✅ PyTorch installed
- ✅ KITTI dataset downloaded (sequences 01, 02, 03, 05, 06, 08)
- ✅ Ground truth poses available
- ✅ `vo/scale_net.py` implemented

**To Install**:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# or for GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

**Let's begin Week 3! 🚀**
