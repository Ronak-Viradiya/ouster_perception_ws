#!/usr/bin/env python3
import argparse
import gc
import os
import sys
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import torchvision.transforms.functional as TF
from tqdm import tqdm
import yaml

SRC_DIR = Path(__file__).resolve().parent.parent  
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

MODEL_REGISTRY = {}

def register_model(name):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

try:
    from models.model_scripts.salsanext import SalsaNext
    register_model('salsanext')(SalsaNext)
except ImportError as e:
    print(f"⚠ SalsaNext not imported: {e}")
    try:
        from models.model_scripts.salsanext import create_model as salsanext_factory
        class SalsaNextWrapper(nn.Module):
            def __init__(self, num_classes, input_channels, height, width, **kwargs):
                super().__init__()
                self.model = salsanext_factory(
                    'salsanext', num_classes=num_classes,
                    input_channels=input_channels, height=height, width=width, **kwargs)
            def forward(self, x):
                return self.model(x)
        register_model('salsanext')(SalsaNextWrapper)
        print("   Using SalsaNext factory function (wrapped).")
    except ImportError as e2:
        print(f"⚠ SalsaNext factory also not found: {e2}")

try:
    from models.model_scripts.rangenetpp import RangeNetPlusPlus
    register_model('rangenetpp')(RangeNetPlusPlus)
except ImportError as e:
    print(f"⚠ RangeNet++ not imported: {e}")
    try:
        from models.model_scripts.rangenetpp import create_model as rangenet_factory
        class RangeNetPPWrapper(nn.Module):
            def __init__(self, num_classes, input_channels, height, width, **kwargs):
                super().__init__()
                self.model = rangenet_factory(num_classes=num_classes,
                                              input_channels=input_channels,
                                              height=height, width=width, **kwargs)
            def forward(self, x):
                return self.model(x)
        register_model('rangenetpp')(RangeNetPPWrapper)
        print("   Using RangeNet++ factory function (wrapped).")
    except ImportError as e2:
        print(f"⚠ RangeNet++ factory also not found: {e2}")

def create_model(model_name: str,
                 num_classes: int,
                 input_channels: int,
                 height: int,
                 width: int,
                 **kwargs) -> nn.Module:
    
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    return MODEL_REGISTRY[model_name](
        num_classes=num_classes,
        input_channels=input_channels,
        height=height,
        width=width,
        **kwargs
    )

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        ce = F.cross_entropy(inputs, targets, reduction='none',
                             ignore_index=self.ignore_index)
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce
        if self.alpha is not None:
            valid = targets != self.ignore_index
            a_t = torch.ones_like(targets, dtype=loss.dtype)
            a_t[valid] = self.alpha[targets[valid]]
            loss = a_t * loss
        return loss.mean()

def augment_data(input_tensor, label_tensor, prob=0.5):
    if random.random() < prob:
        input_tensor = TF.hflip(input_tensor)
        label_tensor = TF.hflip(label_tensor)

    if random.random() < prob:
        angle = random.uniform(-3.0, 3.0)
        input_tensor = TF.rotate(input_tensor, angle,
                                 interpolation=TF.InterpolationMode.BILINEAR)
        label_tensor = TF.rotate(label_tensor.unsqueeze(0), angle,
                                 interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

    if random.random() < 0.3:
        valid_mask = (input_tensor[0:1] > 0).float()
        input_tensor = input_tensor.clone()
        input_tensor[0:1] = torch.clamp(
            input_tensor[0:1] + torch.randn_like(input_tensor[0:1]) * 0.02 * valid_mask,
            0.0, 1.0)
        input_tensor[4:5] = torch.clamp(
            input_tensor[4:5] + torch.randn_like(input_tensor[4:5]) * 0.02 * valid_mask,
            0.0, 1.0)
    return input_tensor, label_tensor

class LiDARDataset(Dataset):
    def __init__(self, data_dir, split='train', num_classes=7,
                 max_range=100.0, max_intensity=65535.0,
                 normalize_inputs=True, augment=False):
        self.data_dir = os.path.join(data_dir, split)
        self.num_classes = num_classes
        self.max_range = max_range
        self.max_intensity = max_intensity
        self.normalize = normalize_inputs
        self.augment = augment and (split == 'train')

        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
        self.files = sorted(f for f in os.listdir(self.data_dir) if f.endswith('.npz'))
        print(f"📂 {split}: {len(self.files)} files in {self.data_dir}")
        if self.augment:
            print("   Augmentation: ON (flip + rotation + noise)")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.data_dir, self.files[idx])
        data = np.load(path)

        range_img = data['range'].astype(np.float32)
        x_img = data['x'].astype(np.float32)
        y_img = data['y'].astype(np.float32)
        z_img = data['z'].astype(np.float32)
        int_img = data['intensity'].astype(np.float32)

        if self.normalize:
            range_img = np.clip(range_img / self.max_range, 0.0, 1.0)
            x_img = np.clip(x_img / self.max_range, -1.0, 1.0)
            y_img = np.clip(y_img / self.max_range, -1.0, 1.0)
            z_img = np.clip(z_img / self.max_range, -1.0, 1.0)
            int_img = np.clip(int_img / self.max_intensity, 0.0, 1.0)

        label = np.clip(data['label'].copy(), 0, self.num_classes - 1)

        input_t = torch.from_numpy(
            np.stack([range_img, x_img, y_img, z_img, int_img], axis=0)).float()
        label_t = torch.from_numpy(label).long()

        if self.augment:
            input_t, label_t = augment_data(input_t, label_t)

        return input_t, label_t


def compute_class_weights(dataset, num_classes, device, class_names):
    counts = Counter()
    print("📊 Computing class frequencies (this may take a minute)...")
    for i in tqdm(range(len(dataset)), desc="Counting labels"):
        _, lbl = dataset[i]
        counts.update(lbl.numpy().flatten().tolist())

    total_labeled = sum(v for k, v in counts.items() if k != 0)
    weights = torch.zeros(num_classes, dtype=torch.float, device=device)
    num_semantic = num_classes - 1

    for cls in range(1, num_classes):
        if counts[cls] > 0:
            weights[cls] = total_labeled / (num_semantic * counts[cls])
        else:
            weights[cls] = 1.0
            print(f"⚠️  Class {cls} ({class_names[cls]}) has zero samples!")

    weights[0] = 0.0 

    semantic = weights[1:]
    if semantic.sum() > 0:
        semantic = semantic / semantic.mean()
        weights[1:] = semantic

    print("\n📊 Final class weights (applied in loss):")
    for cls in range(num_classes):
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        count_str = f"{counts[cls]:,}" if counts[cls] > 0 else "0"
        print(f"   {name:16}: {weights[cls]:.4f}  (pixel count: {count_str})")
    print(f"\n   Total labeled pixels: {total_labeled:,}")
    return weights

def optimizer_step(optimizer, scaler, use_amp):
    if use_amp:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g['params']], 1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g['params']], 1.0)
        optimizer.step()
    optimizer.zero_grad()

def train_one_epoch(model, loader, criterion, optimizer, device, epoch,
                    scaler, accum_steps):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0

    optimizer.zero_grad()
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [train]')
    last_idx = -1

    for batch_idx, (inputs, labels) in enumerate(pbar):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        last_idx = batch_idx

        use_amp = (device.type == 'cuda') and (scaler is not None)
        with autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels) / accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        with torch.no_grad():
            preds = torch.argmax(outputs, dim=1)
            mask = labels != 0
            if mask.sum() > 0:
                correct = ((preds == labels) & mask).sum().item()
                total_pixels += mask.sum().item()
                total_correct += correct
                acc = 100.0 * correct / mask.sum().item()
            else:
                acc = 0.0

        pbar.set_postfix(loss=f'{loss.item()*accum_steps:.4f}', acc=f'{acc:.1f}%')

        if (batch_idx + 1) % accum_steps == 0:
            optimizer_step(optimizer, scaler, use_amp)

        total_loss += loss.item() * accum_steps

    if (last_idx + 1) % accum_steps != 0:
        optimizer_step(optimizer, scaler, use_amp)

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0
    avg_acc = 100.0 * total_correct / total_pixels if total_pixels > 0 else 0.0
    return avg_loss, avg_acc

def validate(model, loader, criterion, device, num_classes, class_names):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Validation'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            total_loss += criterion(outputs, labels).item()

            preds = torch.argmax(outputs, dim=1)
            mask = labels != 0
            if mask.sum() > 0:
                total_correct += ((preds == labels) & mask).sum().item()
                total_pixels += mask.sum().item()
                idx = labels[mask] * num_classes + preds[mask]
                confusion += torch.bincount(idx, minlength=num_classes**2) \
                                   .reshape(num_classes, num_classes)

    avg_loss = total_loss / len(loader) if loader else 0.0
    avg_acc = 100.0 * total_correct / total_pixels if total_pixels > 0 else 0.0

    print("\n📊 Per-class validation accuracy:")
    for cls in range(1, num_classes):
        row = confusion[cls]
        total = row.sum().item()
        correct = confusion[cls, cls].item()
        name = class_names[cls] if cls < len(class_names) else f"class_{cls}"
        acc_pct = 100 * correct / total if total > 0 else 0.0
        print(f"   {name:16}: {acc_pct:.1f}%  ({correct}/{total})")

    return avg_loss, avg_acc, confusion

def main():
    parser = argparse.ArgumentParser(
        description='Train a LiDAR segmentation model (SalsaNext / RangeNet++).')
    parser.add_argument('--config', default='config/params.yaml',
                        help='Path to YAML config file.')
    parser.add_argument('--data_dir', default=None,
                        help='Dataset root containing train/ and val/ folders.')
    parser.add_argument('--sequence', default='00',
                        help='Sequence ID (used if combined/ is missing).')
    parser.add_argument('--model', type=str, default=None,
                        choices=list(MODEL_REGISTRY.keys()) if MODEL_REGISTRY else None,
                        help='Model architecture (overrides config).')
    parser.add_argument('--epochs', type=int, default=50)
    
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size per GPU. Keep low for small GPUs.')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--accum_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * accum_steps).')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--loss', choices=['ce', 'focal'], default='focal')
    parser.add_argument('--no_aug', action='store_true')
    parser.add_argument('--no_norm', action='store_true')
    parser.add_argument('--patience', type=int, default=15)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg.get('model', {})
    num_classes = model_cfg.get('num_classes', 7)
    class_names_dict = model_cfg.get('names', {})
    if isinstance(class_names_dict, dict):
        class_names = [class_names_dict[i] for i in range(num_classes)]
    else:
        class_names = [f"class_{i}" for i in range(num_classes)]

    architecture = args.model or model_cfg.get('architecture', 'salsanext')
    if architecture not in MODEL_REGISTRY:
        print(f"❌ Model '{architecture}' not found in registry. Available: {list(MODEL_REGISTRY.keys())}")
        sys.exit(1)

    input_ch = model_cfg.get('input_channels', 5)
    proj_h = model_cfg.get('height', 128)
    proj_w = model_cfg.get('width', 2048)

    sensor_cfg = cfg.get('sensor', {})
    max_range = sensor_cfg.get('max_range', 100.0)
    max_intensity = sensor_cfg.get('intensity_max', 65535.0)

    paths_cfg = cfg.get('paths', {})
    sequences_dir = paths_cfg.get('sequences_dir', 'data/sequences')

    if args.data_dir:
        data_dir = args.data_dir
    else:
        base = Path(sequences_dir).expanduser().resolve()
        combined = base / 'combined'
        if combined.exists() and ((combined / 'train').is_dir() or (combined / 'val').is_dir()):
            data_dir = str(combined)
        else:
            seq_dir = base / args.sequence / 'training'
            if seq_dir.is_dir():
                data_dir = str(seq_dir)
            else:
                print(f"❌ Training data not found at {combined} or {seq_dir}")
                sys.exit(1)
    print(f"✅ Data directory: {data_dir}")

    model_save_dir = Path(f'models/trained/{architecture}_sequence_{args.sequence}')
    model_save_dir.mkdir(parents=True, exist_ok=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("="*60)
    print(f"🚀 Training {architecture.upper()} on SemanticTHAB ({num_classes} classes)")
    print("="*60)
    print(f"   Config       : {config_path}")
    print(f"   Data         : {data_dir}")
    print(f"   Save dir     : {model_save_dir}")
    print(f"   Device       : {device}")
    print(f"   Classes      : {num_classes}  {class_names}")
    print(f"   Batch        : {args.batch_size} × {args.accum_steps} = effective {args.batch_size*args.accum_steps}")
    print(f"   Epochs       : {args.epochs if args.epochs else model_cfg.get('epochs', 30)}  |  LR: {args.lr}")
    print(f"   Loss         : {args.loss.upper()}  |  Aug: {'OFF' if args.no_aug else 'ON'}")
    print("="*60)

    train_ds = LiDARDataset(
        data_dir, 'train', num_classes,
        max_range=max_range, max_intensity=max_intensity,
        normalize_inputs=not args.no_norm, augment=not args.no_aug
    )
    val_ds = LiDARDataset(
        data_dir, 'val', num_classes,
        max_range=max_range, max_intensity=max_intensity,
        normalize_inputs=not args.no_norm, augment=False
    )

    if len(train_ds) == 0:
        print("❌ No training samples found!"); sys.exit(1)

    class_weights = compute_class_weights(train_ds, num_classes, device, class_names)

    loader_kw = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == 'cuda'),
        persistent_workers=(args.num_workers > 0)
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, **loader_kw) if len(val_ds) > 0 else None

    print(f"\n🔄 Building {architecture} ({input_ch}→{proj_h}×{proj_w}, {num_classes} classes)")
    model = create_model(architecture,
                         num_classes=num_classes,
                         input_channels=input_ch,
                         height=proj_h,
                         width=proj_w).to(device)
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = FocalLoss(gamma=2.0, alpha=class_weights, ignore_index=0) \
                if args.loss == 'focal' else nn.CrossEntropyLoss(weight=class_weights, ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    scaler = GradScaler() if device.type == 'cuda' else None

    epochs = args.epochs if args.epochs is not None else model_cfg.get('epochs', 30)
    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(epochs):
        print(f"\n{'='*40}\nEpoch {epoch+1}/{epochs}\n{'='*40}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            scaler, args.accum_steps
        )

        if val_loader:
            val_loss, val_acc, _ = validate(
                model, val_loader, criterion, device, num_classes, class_names)
            scheduler.step(val_loss)
        else:
            val_loss, val_acc = 0.0, 0.0
            scheduler.step(train_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"\n   Train  loss={train_loss:.4f}  acc={train_acc:.1f}%")
        if val_loader:
            print(f"   Val    loss={val_loss:.4f}  acc={val_acc:.1f}%")
        print(f"   LR     {optimizer.param_groups[0]['lr']:.2e}")

        best_path = model_save_dir / f'{architecture}_best.pth'
        if val_loader and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'num_classes': num_classes,
                'class_names': class_names,
                'architecture': architecture,
                'input_channels': input_ch,
                'height': proj_h,
                'width': proj_w,
                'config': cfg,
            }, best_path)
            print(f"   💾 Best model saved (val_loss={best_val_loss:.4f})")
        elif val_loader and args.patience > 0:
            patience_counter += 1
            print(f"   ⏳ No improvement {patience_counter}/{args.patience} (best={best_val_loss:.4f})")
            if patience_counter >= args.patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'num_classes': num_classes,
                'class_names': class_names,
                'architecture': architecture,
                'config': cfg,
            }, model_save_dir / f'{architecture}_epoch_{epoch+1}.pth')

    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'num_classes': num_classes,
        'class_names': class_names,
        'architecture': architecture,
        'input_channels': input_ch,
        'height': proj_h,
        'width': proj_w,
        'config': cfg,
    }, model_save_dir / f'{architecture}_final.pth')

    print("\n" + "="*60)
    print("🎉 Training complete!")
    print(f"   Best model : {model_save_dir}/{architecture}_best.pth")
    print(f"   Final model: {model_save_dir}/{architecture}_final.pth")
    print(f"   Best train acc : {max(history['train_acc']):.1f}%")
    if val_loader:
        print(f"   Best val acc   : {max(history['val_acc']):.1f}%")
    print("="*60)

if __name__ == '__main__':
    main()