"""Synthetic-only curriculum data adapters for the maintained OpenOCR reader."""
import io
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms as T


def augmentation(seed):
    import albumentations as A
    # Same operations and parameters as model 2/src/parseq/train.py v2.
    return A.Compose([
        A.ColorJitter(brightness=.25, contrast=.25, saturation=.25, hue=.08, p=.7),
        A.OneOf([A.GaussianBlur(blur_limit=(3,5), p=1), A.MotionBlur(blur_limit=5, p=1), A.MedianBlur(blur_limit=3, p=1)], p=.4),
        A.GaussNoise(p=.3), A.ImageCompression(quality_range=(50,95), p=.4),
        A.Affine(rotate=(-3,3), translate_percent={'x':(-.02,.02),'y':(-.04,.04)}, scale=(.95,1.05), shear={'x':(-3,3)}, border_mode=0, fill=255, p=.6),
        A.CoarseDropout(num_holes_range=(1,3), hole_height_range=(2,8), hole_width_range=(2,12), fill=255, p=.3),
    ], seed=seed, strict=True)


def transform():
    return T.Compose([T.Resize((32,128), interpolation=T.InterpolationMode.BICUBIC), T.ToTensor(), T.Normalize(.5,.5)])


class ReaderDataset(Dataset):
    need_reset = False

    def __init__(self, env, indices, charset, seed=None):
        from openrec.preprocess.ar_label_encode import ARLabelEncode
        self.env = env
        self.indices = np.asarray(indices, dtype=np.int64)
        self.encode = ARLabelEncode(max_text_length=25, character_dict_path=str(charset), use_space_char=False)
        self.transform = transform()
        self.aug = augmentation(seed) if seed is not None else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = int(self.indices[index])
        with self.env.begin() as txn:
            data = txn.get(f'image-{index:09d}'.encode())
            label = txn.get(f'label-{index:09d}'.encode()).decode()
        im = Image.open(io.BytesIO(data)).convert('RGB')
        if self.aug is not None:
            im = Image.fromarray(self.aug(image=np.asarray(im))['image'])
        encoded = self.encode({'label':label})
        assert encoded is not None
        return self.transform(im), encoded['label'], encoded['length']


class BucketSampler(Sampler):
    """Fixed-length, replacement sampling; epoch schedule uses one-based epochs."""
    def __init__(self, buckets, seed, start_epoch=1, callback=None):
        self.buckets = np.asarray(buckets)
        self.seed = seed
        self.epoch = start_epoch
        self.callback = callback
        self.groups = {b:np.flatnonzero(self.buckets == b) for b in ['easy','medium','hard']}
        assert all(len(v) for v in self.groups.values())

    def __len__(self):
        return len(self.buckets)

    def __iter__(self):
        epoch = self.epoch
        weights = [1.,0.,0.] if epoch == 1 else [.5,.5,0.] if epoch == 2 else [.15,.25,.60]
        counts = [round(len(self)*w) for w in weights]
        counts[int(np.argmax(weights))] += len(self)-sum(counts)
        rng = np.random.default_rng(self.seed+epoch*100003)
        values = np.concatenate([rng.choice(self.groups[b], n, replace=True) for b,n in zip(self.groups,counts) if n])
        rng.shuffle(values)
        assert len(values) == len(self)
        if self.callback:
            self.callback(epoch, dict(zip(self.groups,counts)))
        self.epoch += 1
        return iter(values.tolist())
