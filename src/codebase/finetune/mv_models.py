import torch
import torch.nn as nn

# Define the VisionLinearProbe class for BIRADS and Density classification
class VisionLinearProbe(nn.Module):
    def __init__(self, vision_model, num_classes):
        super(VisionLinearProbe, self).__init__()
        self.vision_model = vision_model
        self.linear = nn.Linear(vision_model.aggregator.projection_layer.out_features, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, device=None):
        with torch.no_grad():
            images, view_seq = x['images'], x['view_seqs']
            features = self.vision_model(images, view_seq)
        logits = self.linear(features)
        # probabilities = self.softmax(logits)
        return logits
