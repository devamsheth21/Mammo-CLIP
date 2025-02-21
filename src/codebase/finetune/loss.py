import torch
import torch.nn.functional as F
import torch.nn as nn

def group_contrastive_loss(image_embeds, groups, temperature=0.1):
    """
    Args:
        image_embeds: Tensor of shape [batch_size, embed_dim]
        groups: Tensor of shape [batch_size] containing group IDs
    """
    # Normalize embeddings
    image_embeds = F.normalize(image_embeds, dim=-1)
    
    # Compute similarity matrix
    sim_matrix = torch.mm(image_embeds, image_embeds.t()) / temperature
    
    # Create positive mask (same group)
    group_mask = (groups.unsqueeze(1) == groups.unsqueeze(0)).float()
    eye_mask = torch.eye(groups.size(0), device=groups.device)
    positive_mask = group_mask - eye_mask  # Exclude self-similarity
    
    # Compute positive and negative terms
    positives = torch.sum(sim_matrix * positive_mask, dim=1)
    negatives = torch.logsumexp(sim_matrix * (1 - group_mask), dim=1)
    
    # Final loss
    loss = -torch.mean(positives - negatives)
    return loss

    # contrastive loss Function
def contrastive_loss(image_embeddings, text_embeddings, logit_scale):
    logits_per_image = logit_scale * image_embeddings @ text_embeddings.T
    logits_per_text = logits_per_image.T
    labels = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)

    loss_img = nn.CrossEntropyLoss()(logits_per_image, labels)
    loss_txt = nn.CrossEntropyLoss()(logits_per_text, labels)
    return (loss_img + loss_txt) / 2