

from breastclip.model.modules import load_image_encoder, LinearClassifier
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


EMBEDDING_DIM = 96
MAX_VIEWS = 2




class AllImageTransformer(nn.Module):
    def __init__(self, args, image_encoder, image_encoder_type):
        super(AllImageTransformer, self).__init__()
        self.args = args
        self.args.hidden_dim = 512
        self.args.num_layers = 3
        self.args.num_heads = 8
        self.args.num_images = 2 #4
        # self.args.num_classes = n_class
        self.args.dropout = 0.25
        self.image_encoder = image_encoder
        self.args.precomputed_hidden_dim = self.image_encoder.out_dim
        self.image_encoder_type = image_encoder_type


        self.projection_layer = nn.Linear(self.args.precomputed_hidden_dim, self.args.hidden_dim)


        self.mask_embedding = nn.Embedding(2, self.args.precomputed_hidden_dim, padding_idx=1)
        self.kept_images_vec = nn.Parameter(torch.ones([1, self.args.num_images, 1]), requires_grad=False)


        self.transformer = Transformer(self.args)


        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=self.args.dropout)
        # self.fc = nn.Linear(self.args.hidden_dim, self.args.num_classes)


    def mask_input(self, x, view_seq):
        B, N, _ = x.size()
        device = x.device
        mask_prob = 0
        is_mask = torch.bernoulli(self.kept_images_vec.expand([B, N, 1]) * mask_prob).to(device)
        is_mask = is_mask * (view_seq < MAX_VIEWS).unsqueeze(-1).float().to(device)
        is_kept = 1 - is_mask
        x = x * is_kept + self.mask_embedding(is_kept.squeeze(-1).long())
        return x, is_mask


    def forward(self, images, view_seq):
        device = images.device


        batch_size, num_images, C, H, W = images.size()
        images = images.view(batch_size * num_images, C, H, W)


        input_dict = {"image": images, "breast_clip_train_mode": True}
        image_features, _ = self.image_encoder(input_dict)
        image_features = image_features.view(batch_size, num_images, -1)  # (batch_size, 4, 2048)


        x, is_mask = self.mask_input(image_features, view_seq)
        x = self.projection_layer(x)
        # time_seq = time_seq.to(device).long()
        view_seq = view_seq.to(device).long()
        # side_seq = side_seq.to(device).long()


        transformer_hidden = self.transformer(x, view_seq)
        pooled_feature = transformer_hidden.mean(dim=1)
        hidden = self.relu(pooled_feature)
        hidden = self.dropout(hidden)
        # logits = self.fc(hidden)


        return pooled_feature


    def forward_features_only(self, z, view_seq):
        """
        Directly feed the 2,048-d features z into the aggregator pipeline:
        1) mask_input, 2) projection, 3) transformer, 4) fc => logits
        z is shape (B*N, 2048). We'll reshape to (B, N, 2048) inside.
        """
        # Suppose we know B and N
        # Or pass them in as separate args
        device = z.device
        B = view_seq.size(0)
        N = z.size(0) // B


        z = z.view(B, N, -1)  # (B, N, 2048)
        # Now the rest is the same as your original aggregator forward
        # except we skip the line: image_features, _ = self.image_encoder(...)
        x, _ = self.mask_input(z, view_seq)
        x = self.projection_layer(x)  # (B, N, hidden_dim)
        # time_seq = time_seq.to(device).long()
        view_seq = view_seq.to(device).long()
        # side_seq = side_seq.to(device).long()


        transformer_hidden = self.transformer(x, view_seq)
        pooled_feature = transformer_hidden.mean(dim=1)
        hidden = self.relu(pooled_feature)
        hidden = self.dropout(hidden)
        # logits = self.fc(hidden)
        return pooled_feature




class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args


        # embedding_dim_per_type = EMBEDDING_DIM // 3  # 32


        # self.time_embed = nn.Embedding(MAX_TIME + 1, embedding_dim_per_type, padding_idx=0)
        self.view_embed = nn.Embedding(MAX_VIEWS + 1, EMBEDDING_DIM, padding_idx=0)
        # self.side_embed = nn.Embedding(MAX_SIDES + 1, embedding_dim_per_type, padding_idx=0)


        self.embed_fc = nn.Linear(EMBEDDING_DIM, args.hidden_dim)
        self.layers = nn.ModuleList([TransformerLayer(args) for _ in range(args.num_layers)])


    def forward(self, x, view_seq):
        # time_emb = self.time_embed(time_seq)  # (batch_size, num_images, embedding_dim_per_type)
        view_emb = self.view_embed(view_seq)  # (batch_size, num_images, embedding_dim_per_type)
        # side_emb = self.side_embed(side_seq)  # (batch_size, num_images, embedding_dim_per_type)

        embed = view_emb
        # embed = torch.cat([time_emb, view_emb, side_emb], dim=-1)  # (batch_size, num_images, EMBEDDING_DIM)
        embed = self.embed_fc(embed)  # (batch_size, num_images, hidden_dim)


        x = x + embed


        for layer in self.layers:
            x = layer(x)


        return x




class TransformerLayer(nn.Module):
    def __init__(self, args):
        super(TransformerLayer, self).__init__()
        self.args = args
        self.multihead_attention = MultiHead_Attention(args)
        self.layernorm1 = nn.LayerNorm(args.hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.ReLU(),
            nn.Linear(args.hidden_dim, args.hidden_dim)
        )
        self.layernorm2 = nn.LayerNorm(args.hidden_dim)


    def forward(self, x):
        attn_output = self.multihead_attention(x)
        x = self.layernorm1(x + attn_output)
        fc_output = self.fc(x)
        x = self.layernorm2(x + fc_output)
        return x




class MultiHead_Attention(nn.Module):
    def __init__(self, args):
        super(MultiHead_Attention, self).__init__()
        self.args = args
        assert args.hidden_dim % args.num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.dim_per_head = args.hidden_dim // args.num_heads


        self.query = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.key = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.value = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.dropout = nn.Dropout(p=args.dropout)
        self.out_proj = nn.Linear(args.hidden_dim, args.hidden_dim)


    def forward(self, x):
        batch_size, seq_length, hidden_dim = x.size()


        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)


        Q = Q.view(batch_size, seq_length, self.args.num_heads, self.dim_per_head).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.args.num_heads, self.dim_per_head).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.args.num_heads, self.dim_per_head).transpose(1, 2)


        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.dim_per_head)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_length, hidden_dim)
        output = self.out_proj(context)
        return output




class BreastClipMIRAITransformer(nn.Module):
    def __init__(self, args, ckpt):
        super(BreastClipMIRAITransformer, self).__init__()
        print(ckpt["config"]["model"]["image_encoder"])
        self.config = ckpt["config"]["model"]["image_encoder"]
        self.image_encoder = load_image_encoder(ckpt["config"]["model"]["image_encoder"])
        image_encoder_weights = {}
        for k in ckpt["model"].keys():
            if k.startswith("image_encoder."):
                image_encoder_weights[".".join(k.split(".")[1:])] = ckpt["model"][k]
        self.image_encoder.load_state_dict(image_encoder_weights, strict=True)
        self.image_encoder_type = ckpt["config"]["model"]["image_encoder"]["model_type"]
        # self.arch = args.arch.lower()
       
        for param in self.image_encoder.parameters():
            param.requires_grad = False


        self.feature_dim = self.image_encoder.out_dim
        self.aggregator = AllImageTransformer(args=args, image_encoder=self.image_encoder, image_encoder_type=self.image_encoder_type)
        self.raw_features = None
        self.pool_features = None


    def forward(self, images, view_seq):
        return self.aggregator(images, view_seq)
