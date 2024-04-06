"""
@authors: A Bhattacharya, et. al
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the submodules for ViT that were used in the paper "Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance" by Bhattacharya, et. al

@source: https://github.com/git-dhruv/Segformer
"""

import torch.nn as nn

class OverlapPatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding):
        super().__init__()
        self.cn1 = nn.Conv2d(in_channels, out_channels, kernel_size=patch_size, stride = stride, padding = padding)
        self.layerNorm = nn.LayerNorm(out_channels)

    def forward(self, patches):
        """Merge patches to reduce dimensions of input.

        :param patches: tensor with shape (B, C, H, W) where
            B is the Batch size
            C is the number of Channels
            H and W are the Height and Width
        :return: tensor with shape (B, C, H, W)
        """
        x = self.cn1(patches)
        _,_,H, W = x.shape
        x = x.flatten(2).transpose(1,2) #Flatten - (B,C,H*W); transpose B,HW, C
        x = self.layerNorm(x)
        return x,H,W #B, N, EmbedDim
class EfficientSelfAttention(nn.Module):
    def __init__(self, channels, reduction_ratio, num_heads):
        super().__init__()
        assert channels % num_heads == 0, f"channels {channels} should be divided by num_heads {num_heads}."

        self.heads= num_heads

        #### Self Attention Block consists of 2 parts - Reduction and then normal Attention equation of queries and keys###
        
        # Reduction Parameters #
        self.cn1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=reduction_ratio, stride= reduction_ratio)
        self.ln1 = nn.LayerNorm(channels)
        # Attention Parameters #
        self.keyValueExtractor = nn.Linear(channels, channels * 2)
        self.query = nn.Linear(channels, channels)
        self.smax = nn.Softmax(dim=-1)
        self.finalLayer = nn.Linear(channels, channels) 


    def forward(self, x, H, W):

        """ Perform self attention with reduced sequence length

        :param x: tensor of shape (B, N, C) where
            B is the batch size,
            N is the number of queries (equal to H * W)
            C is the number of channels
        :return: tensor of shape (B, N, C)
        """
        B,N,C = x.shape
        # B, N, C -> B, C, N
        x1 = x.clone().permute(0,2,1)
        # BCN -> BCHW
        x1 = x1.reshape(B,C,H,W)
        x1 = self.cn1(x1)
        x1 = x1.reshape(B,C,-1).permute(0,2,1).contiguous()
        x1 = self.ln1(x1)
        # We have got the Reduced Embeddings! We need to extract key and value pairs now
        keyVal = self.keyValueExtractor(x1)
        keyVal = keyVal.reshape(B, -1 , 2, self.heads, int(C/self.heads)).permute(2,0,3,1,4).contiguous()
        k,v = keyVal[0],keyVal[1] #b,heads, n, c/heads
        q = self.query(x).reshape(B, N, self.heads, int(C/self.heads)).permute(0, 2, 1, 3).contiguous()

        dimHead = (C/self.heads)**0.5
        attention = self.smax(q@k.transpose(-2, -1)/dimHead)
        attention = (attention@v).transpose(1,2).reshape(B,N,C)

        x = self.finalLayer(attention) #B,N,C        
        return x

class MixFFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super().__init__()
        expanded_channels = channels*expansion_factor
        #MLP Layer        
        self.mlp1 = nn.Linear(channels, expanded_channels)
        #Depth Wise CNN Layer
        self.depthwise = nn.Conv2d(expanded_channels, expanded_channels, kernel_size=3,  padding='same', groups=channels)
        #GELU
        self.gelu = nn.GELU()
        #MLP to predict
        self.mlp2 = nn.Linear(expanded_channels, channels)

    def forward(self, x, H, W):
        """ Perform self attention with reduced sequence length

        :param x: tensor with shape (B, C, H, W) where
            B is the Batch size
            C is the number of Channels
            H and W are the Height and Width
        :return: tensor with shape (B, C, H, W)
        """
        # Input BNC instead of BCHW
        # BNC -> B,N,C*exp 
        x = self.mlp1(x)
        B,N,C = x.shape
        # Prepare for the CNN operation, channel should be 1st dim
        # B,N, C*exp -> B, C*exp, H, W 
        x = x.transpose(1,2).view(B,C,H,W)

        #Depth Conv - B, N, Cexp 
        x = self.gelu(self.depthwise(x).flatten(2).transpose(1,2))

        #Back to the orignal shape
        x = self.mlp2(x) # BNC
        return x

class MixTransformerEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size, stride, padding, 
                 n_layers, reduction_ratio, num_heads, expansion_factor):
        super().__init__()
        self.patchMerge = OverlapPatchMerging(in_channels, out_channels, patch_size, stride, padding) # B N embed dim
        #You might be wondering why I didn't used a cleaner implementation but the input to each forward function is different
        self._attn = nn.ModuleList([EfficientSelfAttention(out_channels, reduction_ratio, num_heads) for _ in range(n_layers)])
        self._ffn = nn.ModuleList([MixFFN(out_channels,expansion_factor) for _ in range(n_layers)])
        self._lNorm = nn.ModuleList([nn.LayerNorm(out_channels) for _ in range(n_layers)])

    def forward(self, x):
        """ Run one block of the mix vision transformer

        :param x: tensor with shape (B, C, H, W) where
            B is the Batch size
            C is the number of Channels
            H and W are the Height and Width
        :return: tensor with shape (B, C, H, W)
        """
        B,C,H,W = x.shape
        x,H,W = self.patchMerge(x) # B N embed dim (C)
        for i in range(len(self._attn)):
            x = x + self._attn[i].forward(x, H, W) #BNC
            x = x + self._ffn[i].forward(x, H, W) #BNC
            x = self._lNorm[i].forward(x) #BNC
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous() #BCHW
        return x
