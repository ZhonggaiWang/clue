"""RGB-only VOC data. Incremental training never opens segmentation masks."""
from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter
import tasks


class VOCDataset(Dataset):
    def __init__(self, root, list_folder, task, step, train=True, crop_size=448,
                 scales=(0.5, 2.0), limit=0, seg_label_dir=None):
        self.root, self.list_folder = Path(root), Path(list_folder)
        self.step, self.train, self.crop_size = step, train, crop_size
        self.seg_label_dir = Path(seg_label_dir) if seg_label_dir else self.root / "SegmentationClassAug"
        self.scales = scales
        self.current, self.old, _ = tasks.get_task_labels('voc', task, step)
        self.seen = self.old + self.current
        split = 'train' if train else 'val'
        path = self.list_folder / 'incremental_split' / f'{split}_{task}_step_{step+1}.txt'
        self.names = [line.split()[0] for line in path.read_text().splitlines() if line.strip()]
        if limit:
            self.names = self.names[:limit]
        self.cls_labels = np.load(self.list_folder / 'cls_labels_onehot.npy', allow_pickle=True).item() if train else None
        self.jitter = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)
        self.label_map = np.full(256, 255, dtype=np.int64)
        for i, c in enumerate(self.seen):
            self.label_map[c] = i
        if self.train and self.step == 0:
            for c in range(1,21):
                if c not in self.seen: self.label_map[c] = 0
        self.label_map[255] = 255

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        image = Image.open(self.root / 'JPEGImages' / (name + '.jpg')).convert('RGB')
        # Dense labels are used ONLY at step 0 or for held-out evaluation.
        mask = None
        if not self.train or self.step == 0:
            path = (self.seg_label_dir if self.train else self.root / 'SegmentationClass') / (name + '.png')
            if not path.exists():
                path = self.root / 'SegmentationClass' / (name + '.png')
            with Image.open(path) as f:
                mask = np.array(f, dtype=np.uint8)
            if mask.ndim != 2:
                raise ValueError(f'Expected indexed VOC mask: {path}')
        if self.train:
            ratio = random.uniform(*self.scales)
            size = tuple(max(1, round(s * ratio)) for s in image.size)
            image = image.resize(size, Image.Resampling.BILINEAR)
            if mask is not None:
                mask = np.array(Image.fromarray(mask).resize(size, Image.Resampling.NEAREST))
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if mask is not None:
                    mask = mask[:, ::-1].copy()
            image = self.jitter(image)
        rgb = np.asarray(image, dtype=np.float32)
        h, w = rgb.shape[:2]
        valid = np.ones((h, w), dtype=np.bool_)
        if self.train:
            c = self.crop_size
            padded = np.empty((max(c, h), max(c, w), 3), dtype=np.float32)
            padded[...] = [123.675, 116.28, 103.53]
            padded[:h, :w] = rgb
            padded_valid = np.zeros(padded.shape[:2], dtype=np.bool_)
            padded_valid[:h, :w] = True
            y, x = random.randint(0, max(0, h-c)), random.randint(0, max(0, w-c))
            rgb = padded[y:y+c, x:x+c]
            valid = padded_valid[y:y+c, x:x+c]
            if mask is not None:
                padded_mask = np.full(padded.shape[:2], 255, dtype=np.uint8)
                padded_mask[:h, :w] = mask
                mask = padded_mask[y:y+c, x:x+c]
        image = (rgb / 255.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        out = dict(name=name, image=torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32)),
                   valid=torch.from_numpy(valid.copy()))
        if self.train:
            tags = np.zeros(len(self.seen)-1, dtype=np.float32)
            current = set(self.current)
            for i, c in enumerate(self.seen[1:]):
                if c in current:
                    tags[i] = self.cls_labels[name][c-1]
            out['tags'] = torch.from_numpy(tags)
        if mask is not None:
            out['mask'] = torch.from_numpy(self.label_map[mask].copy())
        return out
