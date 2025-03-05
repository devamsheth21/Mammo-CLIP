import sys
sys.path.append('..')
import torch
from breastclip.model import build_model
from breastclip.model.modules import load_image_encoder, load_projection_head, load_text_encoder
from breastclip.data.data_utils import load_tokenizer
from multiview_model import BreastClipMIRAITransformer
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F
from types import SimpleNamespace

class MultiViewModel(nn.Module):
    def __init__(self, image_model, text_encoder, text_projection, logit_scale, text_pooling):
        super().__init__()
        self.image_model = image_model  #  BreastMIRAITransformer
        self.text_encoder = text_encoder
        self.text_projection = text_projection
        self.logit_scale = logit_scale
        self.text_pooling = text_pooling
        # Freeze text encoder
        for param in self.text_encoder.parameters():
            param.requires_grad = False

    def forward(self, batch, device=None):
        # device = batch["images"].device if device is None else device
        
        # Image embeddings (using both views)
        image_embeddings = self.image_model(
            images=batch["images"],
            view_seq=batch["view_seqs"]
        )
        text_tokens = batch["text_tokens"]
        text_features = self.text_encoder(text_tokens)

        if self.text_pooling == "eos":
            # take features from the eot embedding (eos_token is the highest number in each sequence)
            eos_token_indices = text_tokens["attention_mask"].sum(dim=-1) - 1
            text_features = text_features[torch.arange(text_features.shape[0]), eos_token_indices]
        elif self.text_pooling == "bos":
            text_features = text_features[:, 0]
        elif self.text_pooling == "mean":
            input_mask_expanded = text_tokens["attention_mask"].unsqueeze(axis=-1).expand(text_features.size()).float()
            text_features = torch.sum(text_features * input_mask_expanded, axis=1) / torch.clamp(
                input_mask_expanded.sum(axis=1), min=1e-9)
        # Text embeddings
        # text_features = self.text_encoder(batch["text_tokens"].to(device))
        text_embeddings = self.text_projection(text_features)
        # Normalize features
        image_embeddings = F.normalize(image_embeddings, dim=1) # P= 2 WHY? REMOVE ? 
        text_embeddings = F.normalize(text_embeddings, dim=1)
        return {
            "image_embeddings": image_embeddings,
            "text_embeddings": text_embeddings,
            "labels": torch.arange(image_embeddings.shape[0], device=device),
            "logit_scale": self.logit_scale.exp()
        }

def load_model(config):
    # Load checkpoint
    clip_chk_pt_path = "/mnt/PURENFS/SalkowskiPreprocessedBreast/code/MammoCLIP/checkpoints/b5-model-best-epoch-7.tar"
    ckpt = torch.load(clip_chk_pt_path, map_location="cpu")
    # device = config['device']
    args = SimpleNamespace(**config['model_params'])
    modelname = config['modelname']

    cache_dir = '/mnt/PURENFS/SalkowskiPreprocessedBreast/code/MammoCLIP/hub/'
    # Update config paths
    ckpt["config"]["tokenizer"]['cache_dir'] = cache_dir
    ckpt["config"]["model"]["text_encoder"]["cache_dir"] = cache_dir
    ckpt["config"]["model"]["text_encoder"]["name"] = os.path.join(cache_dir,"models--emilyalsentzer--Bio_ClinicalBERT/")
    model_config = ckpt["config"]["model"]
    
    # Load tokenizer
    tokenizer = load_tokenizer(**ckpt["config"]["tokenizer"])
    
    if modelname.lower() == "mammoclip" :
        # Load Model Standard MammoCLIP
        model = build_model(
            model_config=ckpt["config"]["model"],
            loss_config=ckpt["config"]["loss"],
            tokenizer=tokenizer,
        )
        model.load_state_dict(ckpt["model"], strict=False)
        # Freeze everything except the vision encoder
        for param in model.text_encoder.parameters():
            param.requires_grad = False
        for param in model.text_projection.parameters():
            param.requires_grad = False
        for param in model.image_projection.parameters():
            param.requires_grad = False 
         
    elif modelname.lower() == "multiviewmammoclip":
        # Load image model
        image_model = BreastClipMIRAITransformer(
            args=args,
            ckpt=ckpt,
        )
        
        # Load text components from original model
        text_encoder = load_text_encoder(
            model_config["text_encoder"],
            vocab_size=tokenizer.vocab_size
        )
        # Load projections and logit scale
        text_projection = load_projection_head(
                    embedding_dim=text_encoder.out_dim, config_projection_head= model_config["projection_head"]
                )
        logit_scale = nn.Parameter(ckpt['model']['logit_scale'])
        temperature = model_config["temperature"] if "temperature" in model_config else None
        if temperature:
            logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
        else:
            logit_scale = torch.tensor(1, dtype=torch.float32)
        model =  MultiViewModel(
            image_model=image_model,
            text_encoder=text_encoder,
            text_projection=text_projection,
            logit_scale=logit_scale,
            text_pooling = model_config["text_encoder"]["pooling"]
            )
    else:
        print(f"Model name : {modelname} is invalid")

    return model, tokenizer

def build_multiview_model(ckpt, tokenizer, args):
    cache_dir = '/mnt/PURENFS/SalkowskiPreprocessedBreast/code/MammoCLIP/hub/'
    # Update config paths
    ckpt["config"]["tokenizer"]['cache_dir'] = cache_dir
    ckpt["config"]["model"]["text_encoder"]["cache_dir"] = cache_dir
    ckpt["config"]["model"]["text_encoder"]["name"] = os.path.join(cache_dir,"models--emilyalsentzer--Bio_ClinicalBERT/")
    model_config = ckpt["config"]["model"]
    # Load image model
    image_model = BreastClipMIRAITransformer(
        args=args,
        ckpt=ckpt,
    )
    
    # Load tokenizer
    # tokenizer = load_tokenizer(**ckpt["config"]["tokenizer"])

    # Load text components from original model
    text_encoder = load_text_encoder(
        model_config["text_encoder"],
        vocab_size=tokenizer.vocab_size
    )
    
    # Load projections and logit scale
    text_projection = load_projection_head(
                embedding_dim=text_encoder.out_dim, config_projection_head= model_config["projection_head"]
            )
    logit_scale = nn.Parameter(ckpt['model']['logit_scale'])
    temperature = model_config["temperature"] if "temperature" in model_config else None
    if temperature:
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temperature))
    else:
        logit_scale = torch.tensor(1, dtype=torch.float32)
    return MultiViewModel(
        image_model=image_model,
        text_encoder=text_encoder,
        text_projection=text_projection,
        logit_scale=logit_scale,
        text_pooling = model_config["text_encoder"]["pooling"]
    )