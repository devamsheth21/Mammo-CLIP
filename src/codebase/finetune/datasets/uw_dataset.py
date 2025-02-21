import json
import os
import random
import math
from tqdm.auto import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
from .selective_Sampling import SelectiveSampling2
import sys
sys.path.insert(0,'/mnt/PURENFS/SalkowskiPreprocessedBreast/code/ALBEF/')
from dataset.utils import pre_caption, scale_0_1
from dataset.selective_sampling import SelectiveSampling
import pandas as pd
from torchvision import transforms
class ft_uw_individual(Dataset):
    def __init__(self, ann_file, transform, tokenizer, max_words=500, select = False , test=False, split = "train"):
        self.ann = pd.read_csv(ann_file)
        self.transform = transform
        self.max_words = max_words
        self.all_groups =[]
        self.tokenizer = tokenizer
        self.augment = None #img_R_s.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        if test:
            print("Testing for fraction of data")
            self.ann = self.ann.sample(frac=0.001, random_state=42)
        if split:
            self.ann = self.ann[self.ann["split"] == split]
        if select :
            self.selective_sampler = SelectiveSampling2(self.ann.to_dict('records'))
        
    def __len__(self):
        return len(self.ann)
    
    def __loadimagepair(self, item):
        if item['Missing']=='R':
            imageL = Image.open(item['processed_filepath_L']).convert('RGB')
            imageR = imageL.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        elif item['Missing'] == 'L':
            imageR = Image.open(item['processed_filepath_R']).convert('RGB').transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
            imageL = imageR.transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        else :
            imageL = Image.open(item['processed_filepath_L']).convert('RGB')
            imageR = Image.open(item['processed_filepath_R']).convert('RGB').transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        
        return self.transform(imageL),self.transform(imageR)
    
    def __getitem__(self, index):
        item = self.ann.iloc[index]
        imageL,imageR = self.__loadimagepair(item)
        # Stack images: (2, C, H, W)
        images = torch.stack([imageL, imageR], dim=0)    
        # View sequence (e.g., [0, 1] for L/R)
        view_seq = torch.tensor([0,1], dtype=torch.long)
        caption = pre_caption(item['findings'], self.max_words) 
        group_id = item['group_id']
        accession = item['AccessionNumber']
        return {
            "image": images,
            "text": caption,
            "label": group_id,
            "view_seq":view_seq,
            "acc": accession,
        }

    def collate_fn(self, instances):
        images = torch.stack([ins["image"] for ins in instances], dim=0) # B,2,C,H,W
        texts = [ins["text"] for ins in instances] #B,
        view_seqs = torch.stack([ins["view_seq"] for ins in instances], dim=0)  # (B, 2)
        accessions = [ins['acc'] for ins in instances]
        text_tokens = self.tokenizer(
            texts, padding="max_length", truncation=True, return_tensors="pt", max_length=256
        )

        labels = [ins["label"] for ins in instances]

        return {
            "images": images,
            "acc": accessions,
            "labels": labels,
            "text_tokens": text_tokens,
            "view_seqs" : view_seqs
        }

    def shuffle(self, bs=8, rare_grp_ratio=0.375, batch_shuffle=False):
        self.ann = self.selective_sampling.shuffle(bs=bs, rare_grp_ratio=rare_grp_ratio, batch_shuffle=batch_shuffle)
            

class ft_train_dataset_new(Dataset):
    def __init__(self, ann_file, transform, image_root,tokenizer, max_words=30, num_samples=20, select= None):
        self.ann = pd.read_json(ann_file)
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.all_groups = []
        self.img_ids = {} 
        self.tokenizer = tokenizer
        self.augment = transforms.Compose([
            transforms.RandomResizedCrop(512),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor()
        ])
        # Track group sizes
        self.ann['count'] = self.ann.groupby('group_id')['group_id'].transform('count')
        if select:
            self.selective_sampling = SelectiveSampling(data = self.ann)
        else:
            self.selective_sampler = SelectiveSampling2(self.ann.to_dict('records'))
        # self.selective_sampling = SelectiveSampling(data = self.ann)
        # self.ann['count'] = self.ann['group_id'].map(self.ann['group_id'].value_counts())
    def __len__(self):
        return len(self.ann)
    def __getitem__(self, index):
        item = self.ann.iloc[index]
        image = Image.open(os.path.join(self.image_root, item['image'])).convert('RGB')
        group_size = item['count']
        
        # Return two views for single-instance groups
        if group_size == 1:
            return {
                "view1": self.transform(image),
                "view2": self.augment(image),  # Augmentation happens later
                "text": pre_caption(item['caption'], self.max_words),
                "group_id": item['group_id'],
                "index": item['image_id']#
            }
        else:
            return {
                "image": self.transform(image),
                "text": pre_caption(item['caption'], self.max_words),
                "group_id": item['group_id'],
                "index": item['image_id'] #
            }

    def collate_fn(self, instances):
        """Handle multi-view samples for contrastive learning"""
        images = []
        texts = []
        groups = []
        indices = []
        for ins in instances:
            if 'view1' in ins:  # Single-instance group
                images.extend([ins['view1'], ins['view2']])
                texts.extend([ins['text'], ins['text']])
                groups.extend([ins['group_id'], ins['group_id']])
                indices.extend([ins['index'],ins['index']])
            else:  # Multi-instance group
                images.append(ins['image'])
                texts.append(ins['text'])
                groups.append(ins['group_id'])
                indices.append(ins['index'])
        # Tokenize text
        text_tokens = self.tokenizer(
            texts, padding="max_length", truncation=True, 
            return_tensors="pt", max_length=self.max_words
        )
        
        return {
            "images": torch.stack(images),
            "text_tokens": text_tokens,
            "groups": torch.tensor(groups),
            "index" :indices #
        }

    def shuffle(self, bs=8, rare_grp_ratio=0.375, batch_shuffle=False):
        self.ann = pd.DataFrame(
            self.selective_sampler.shuffle(
                batch_size=bs, 
                rare_ratio=rare_grp_ratio,
                batch_shuffle=batch_shuffle
            )
        ) 
    # def __getitem__(self, index):    
        
    #     ann = self.ann.iloc[index]
        
    #     image_path = os.path.join(self.image_root,ann['image'])        
    #     image = Image.open(image_path).convert('RGB')   
    #     image = self.transform(image)
    #     caption = pre_caption(ann['caption'], self.max_words) 
    #     group_id = ann['group_id']
    #     if ann['count'] ==1:
    #         return {
    #         "view1": image,
    #         "view2" : self.augment(image),
    #         "text": caption,
    #         "label": torch.tensor(index, dtype=torch.long),
    #         "group_id": group_id
    #     }
    #     else:
    #         return {
    #         "image": image,
    #         "text": caption,
    #         "label": torch.tensor(index, dtype=torch.long),
    #         "group_id": group_id
    #         }

    # def collate_fn(self, instances):
    #     # Need to change to handle View
    #     images = torch.stack([ins["image"] for ins in instances], dim=0)
    #     texts = [ins["text"] for ins in instances]

    #     # Tokenize text
    #     text_tokens = self.tokenizer(
    #         texts, padding="max_length", truncation=True, return_tensors="pt", max_length=30
    #     )

    #     labels = torch.stack([ins["label"] for ins in instances], dim=0)
    #     groups = torch.stack([ins["group_id"] for ins in instances], dim=0)
    #     return {
    #         "images": images,
    #         "texts": texts,
    #         "labels": labels,
    #         "text_tokens": text_tokens,
    #         "groups" : groups,
    #     }

    # def shuffle(self, bs=8, rare_grp_ratio=0.375, batch_shuffle=False):
    #     #function to prepare minibatches based on selective sampling strategy
    #     # calls shuffle function from the base class (SelectiveSampling) 
    #     self.ann = self.selective_sampling.shuffle(bs=bs, rare_grp_ratio=rare_grp_ratio, batch_shuffle=batch_shuffle)
    

class ft_train_dataset(Dataset):
    def __init__(self, ann_file, transform, image_root,tokenizer, max_words=30, num_samples=20):
        self.ann = []
        if isinstance(ann_file,list):
            for f in ann_file:
                self.ann += json.load(open(f,'r'))
        else:
            self.ann = json.load(open(ann_file,'r'))

        #this class has selective sampling feature available, can be used when needed by calling SelectiveSampling.shuffle(batch_size)
        self.selective_sampling = SelectiveSampling(data = self.ann)

        #choose only a subset of samples, added to run on small set to make sure everything works fine
        # self.ann = self.ann[:num_samples] 
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words
        self.all_groups = []
        self.img_ids = {}   
        self.tokenizer = tokenizer
        
        n = 0
        for ann in tqdm(self.ann):
            img_id = ann['image_id']
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        print(f"Dataset size:{len(self.ann)}")
        
    def __len__(self):
        return len(self.ann)
    
    def __getitem__(self, index):    
        
        ann = self.ann[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        caption = pre_caption(ann['caption'], self.max_words) 
        group_id = ann['group_id']
        
        return {
            "image": image,
            "text": caption,
            "label": torch.tensor(self.img_ids[ann["image_id"]], dtype=torch.long),
        }
    def collate_fn(self, instances):
        images = torch.stack([ins["image"] for ins in instances], dim=0)
        texts = [ins["text"] for ins in instances]

        # Tokenize text
        text_tokens = self.tokenizer(
            texts, padding="max_length", truncation=True, return_tensors="pt", max_length=30
        )

        labels = torch.stack([ins["label"] for ins in instances], dim=0)

        return {
            "images": images,
            "texts": texts,
            "labels": labels,
            "text_tokens": text_tokens,
        }

    def shuffle(self, bs=8, rare_grp_ratio=0.375, batch_shuffle=False):
        #function to prepare minibatches based on selective sampling strategy
        # calls shuffle function from the base class (SelectiveSampling) 
        self.ann = self.selective_sampling.shuffle(bs=bs, rare_grp_ratio=rare_grp_ratio, batch_shuffle=batch_shuffle)
    