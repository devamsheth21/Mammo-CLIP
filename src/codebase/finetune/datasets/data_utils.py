
import torch
from .uw_dataset import ft_train_dataset_new,ft_uw_individual
import sys
sys.path.insert(0,'/mnt/PURENFS/SalkowskiPreprocessedBreast/code/ALBEF')
from torch.utils.data import DataLoader
from dataset import create_dataset
from breastclip.data.data_utils import load_tokenizer
from torchvision import transforms

def load_dataloader(config,tokenizer, split="train"):
    dataname = config.get('dataset_name','')    
    if 'ft' in dataname:
        mean = config['normalize']['mean']
        std = config['normalize']['std']
        image_size = config.get('image_size',False)
        image_size = (image_size['width'],image_size['height'])
        ft_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((mean, mean, mean), (std, std, std)),
            lambda x: transforms.Resize(image_size)(x) if image_size else x
            ])
    if dataname == 'ft_mv':
        train_dataset = ft_uw_individual(config['train_file'],transform=ft_transform, tokenizer=tokenizer, max_words=config['max_words'], select = config['selective_sampling'],split=split)              
    elif dataname == 'ft':
        train_dataset = ft_train_dataset_new(config['train_file'], ft_transform, config['image_root'],
        max_words=config['max_words'], tokenizer = tokenizer, select= config['selective_sampling']) 
    else : 
        print(f'invalid dataname {dataname}')
        # train_dataset = create_dataset("ft", config)

    return DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        drop_last= True if split=='train' else False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=train_dataset.collate_fn,
    )
