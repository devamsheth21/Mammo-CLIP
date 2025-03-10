
import torch
from .uw_dataset import ft_train_dataset_group,ft_uw_individual, ft_uw_linear_probe
import sys
sys.path.insert(0,'/mnt/PURENFS/SalkowskiPreprocessedBreast/code/ALBEF')
sys.path.append('/mnt/PURENFS/SalkowskiPreprocessedBreast/code/MammoCLIP/Mammo-CLIP/src/codebase/')
from torch.utils.data import DataLoader
from dataset import create_dataset
from breastclip.data.data_utils import load_tokenizer
from torchvision import transforms

def load_dataloader(config, tokenizer=None, split="train"):
    dataname = config.get('dataset_name','')    
    # if 'ft' in dataname:
    mean = config['normalize']['mean']
    std = config['normalize']['std']
    image_size = config.get('image_size',False)
    image_size = (image_size['width'],image_size['height'])
    ft_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean, mean, mean), (std, std, std)),
        transforms.Resize(image_size) if image_size else transforms.Lambda(lambda x: x),
        ])
    if dataname == 'ft_mv':
        train_dataset = ft_uw_individual(
            config['train_file'],
            transform=ft_transform, 
            tokenizer=tokenizer, max_words=config['max_words'],
            select = config['selective_sampling'],
            split=split,
            test=config.get('test',False)
        )              
    elif dataname == 'ft':
        train_dataset = ft_train_dataset_group(
            config['train_file'], 
            ft_transform, 
            config['image_root'],
            max_words=config['max_words'], 
            tokenizer = tokenizer, 
            select= config['selective_sampling']
            )
    elif dataname == 'lp_mv':
        train_dataset = ft_uw_linear_probe(
            config['train_file'],
            transform=ft_transform,
            split=split,
            test=config.get('test',False),
            classname = config.get('classname',None)
        )
    else : 
        print(f'invalid dataname {dataname}')
        # train_dataset = create_dataset("ft", config)

    return DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last= True if split=='train' else False,
        num_workers=8,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=1,
        collate_fn=train_dataset.collate_fn,
    )
