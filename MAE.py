import torch
import timm
import numpy as np
import torch.nn as nn
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block

def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)  # Shuffle the indexes randomly
    backward_indexes = np.argsort(forward_indexes)  # Get the original index positions for easy restoration
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio 

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape  # length, batch, dim
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)  # Randomly shuffle the patches
        patches = patches[:remain_T]  # Get the unmasked patches [T*0.25, B, C]

        return patches, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 emb_dim=768,
                 num_layer=12,
                 num_head=12,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((14 * 14), 1, emb_dim))
        
        # Shuffle and mask the patches
        self.shuffle = PatchShuffle(mask_ratio)
        
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])  # vit-base
        
        # Layer norm for ViT
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()
        
    # Initialize class tokens and positional embeddings
    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img, shuffle_and_mask=True):
        B, C, H, W = img.shape
        patches = rearrange(img, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        if shuffle_and_mask:
            patches, forward_indexes, backward_indexes = self.shuffle(patches)
        else:
            forward_indexes = backward_indexes = None

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes


class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 patch_size=2,
                 emb_dim=768,
                 num_layer=4,
                 num_head=12,
                 target_size=128,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((14 * 14) + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(emb_dim, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=8, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, kernel_size=8, stride=2, padding=1),
        )
        self.AdaptiveAvgPool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.Linear = nn.Linear(768,50)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        if backward_indexes != None:  # If shuffle and mask were applied
            backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
            features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
            features = take_indexes(features, backward_indexes)
            features = features + self.pos_embedding  # Add positional embedding information
        else:  # If shuffle and mask were not applied
            features = features + self.pos_embedding  # Add positional embedding information
        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features) 
        features = rearrange(features, 'b t c -> t b c') 
        cls_token = features[0]  # Global information
        features = features[1:]  # Remove global feature to get image information

        features = rearrange(features, 't b c -> b c t')

        patches = features.view(features.size(0), features.size(1), 14, 14)
        img = self.head(patches)  # Use the head to obtain the reconstructed image
        feature_gap = self.AdaptiveAvgPool2d(patches).squeeze()
        feature_gap = self.Linear(feature_gap)
        return features, img, feature_gap, cls_token


class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 emb_dim=768,
                 encoder_layer=12,
                 encoder_head=12,
                 decoder_layer=4,
                 decoder_head=12,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()
        self.encoder = MAE_Encoder(emb_dim=emb_dim, num_layer=encoder_layer, num_head=encoder_head, mask_ratio=mask_ratio)
        self.decoder = MAE_Decoder(emb_dim=emb_dim, num_layer=decoder_layer, num_head=decoder_head)

    def forward(self, features, shuffle_and_mask=True):
        features, backward_indexes = self.encoder(features, shuffle_and_mask)
        predicted_img = self.decoder(features, backward_indexes)
        return predicted_img

class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder: MAE_Encoder, num_classes=13) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

    def forward(self, features):
        patches = rearrange(features, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits

if __name__ == '__main__':
    img = torch.rand(2, 768, 14, 14)  # Input features (batchsize, 768, 14, 14)
    model = MAE_ViT()
    
    features, predicted_img, mmd_feature = model(img)

    print(predicted_img.shape)  # Output image (batchsize, 3, 128, 128)