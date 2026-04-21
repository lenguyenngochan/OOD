import sys
import os
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import transforms, datasets
import numpy as np
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
sptnet_path = os.path.join(current_dir, '..', 'SPTNet')
sys.path.append(sptnet_path)

from SPTNet.data.get_datasets import get_datasets, get_class_splits
from SPTNet.data.augmentations import get_transform
from SPTNet.model import ContrastiveLearningViewGenerator

def create_loaders(args, val_split=0.2, seed=42):
    train_transform, test_transform = get_transform(args.transform_type, args.image_size, args)
    train_transform = ContrastiveLearningViewGenerator(train_transform, n_views=args.n_views)
    train_dataset, test_dataset, _, _ = get_datasets(args.dataset_name, train_transform, test_transform, args)
    
    labelled_dataset = train_dataset.labelled_dataset
    unlabelled_dataset = train_dataset.unlabelled_dataset
    
    num_labelled = len(labelled_dataset)
    val_size = int(val_split * num_labelled)
    train_labelled_size = num_labelled - val_size
    torch.manual_seed(seed)
    train_labelled_subset, val_subset = random_split(labelled_dataset, [train_labelled_size, val_size])
    
    labelled_train_loader = DataLoader(train_labelled_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    return labelled_train_loader, val_loader, test_loader

def create_ood_test_loader(ood_config, transform, batch_size=32, num_workers=4):
    ood_loaders = {}
    for name, cfg in ood_config.items():
        print(f"Loading OOD dataset: {name}")
        root = cfg['root']
        download = cfg.get('download', False)
        
        if name == 'svhn':
            dataset = datasets.SVHN(root=root, split='test', transform=transform, download=download)
        elif name == 'lsun_resize':
            dataset = datasets.LSUN(root=root, classes=['test'], transform=transform)
        elif name == 'lsun_crop':
            dataset = datasets.LSUN(root=root, classes=['val'], transform=transform)  
        elif name == 'isun':
            class ISUN(torch.utils.data.Dataset):
                def __init__(self, root, transform=None):
                    self.root = root
                    self.transform = transform
                    self.samples = []
                    
                    for fname in os.listdir(root):
                        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                            self.samples.append(os.path.join(root, fname))
                
                def __len__(self):
                    return len(self.samples)
                
                def __getitem__(self, idx):
                    img_path = self.samples[idx]
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    return image, 0
            dataset = ISUN(root=root, transform=transform)
        elif name == 'dtd':
            dataset = datasets.DTD(root=root, split='test', transform=transform, download=download)
        elif name == 'places365':
            dataset = datasets.Places365(root=root, split='val', transform=transform, download=download)
        else:
            raise ValueError(f"Unsupported OOD dataset: {name}")
        
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        ood_loaders[name] = loader
        print(f"Loaded {len(dataset)} samples for OOD dataset: {name}")
    return ood_loaders