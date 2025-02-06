import torch
from torch import nn

# Pos_encoding for Single time integer t
def _pos_encoding(t, output_dim, device = 'cpu'):
    D = output_dim
    v = torch.zeros((D,), device = device)

    i = torch.arange(0, D, device = device)
    div_term = 10000 ** (i / D)

    v[0::2] = torch.sin(t / div_term[0::2])
    v[1::2] = torch.cos(t / div_term[1::2])

    return v

# Positional encoding for batch
def pos_encoding(ts, output_dim, device = 'cpu'):
    # Batch size
    N = len(ts)
    D = output_dim
    v = torch.zeros((N, D), device = device)

    for i in range(N): 
        v[i] = _pos_encoding(ts[i], output_dim, device)

    return v # (N, D) -> positional encoding for each data in batch N * (t integer -> D dimensional vector)

# UNet Conv block 
class Convblock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding = 1),
            nn.BatchNorm2d(out_ch), # Channel-wise standaradization
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding =1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )
    
    def forward(self, x, t):
        N, C, _, _ = x.shape 
        t = self.mlp(t)
        t = t.view(N,C,1,1)
        y = self.convs(x + t)

        return y
    
# UNetConvd
class UNetCond(nn.Module):
    def __init__(self, in_ch = 1, time_embed_dim = 100, num_labels = None):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = Convblock(in_ch, 64, time_embed_dim)
        self.down2 = Convblock(64, 128, time_embed_dim)
        self.bot1 = Convblock(128, 256, time_embed_dim)
        self.up2 = Convblock(128 + 256, 128, time_embed_dim)
        self.up1 = Convblock(64 + 128, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.downsample = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor = 2, mode = 'bilinear')

        # Embedding layer for labels -> This label embedding will be added to time embedding vector
        if num_labels is not None:
            # time embedding vector랑 더해줘야 하니까 label embedding도 차원을 맞춰줌!
            # nn.Embedding is a learnable matrix (num_labels, embedding_dimension)
            self.label_emb = nn.Embedding(num_labels, time_embed_dim) # Initialized randomly(Gaussian..etc) and will be updated -> (num_labels, embedding vector size)

    def forward(self, x, t, labels = None):
        # x -> (N,C,H,W), t -> (N,)
        # Time positional encoding
        v = pos_encoding(t, self.time_embed_dim) # (N,D) D is Time embedding dimension

        # Label
        if labels is not None:
            # [1,2,5] 이면 label 1,2,5에 해당하는 백터를 embedding matrix에서 뽑아온다.
            label_emb = self.label_emb(labels) # (N,D) -> time_emb_dim의 백터 차원을 가지는 label embeddings -> extract the vectors from embedding matrix using indexing 
            v += label_emb

        x1 = self.down1(x,v)
        x = self.downsample(x1)
        x2 = self.down2(x, v)
        x = self.downsample(x2)

        x = self.bot1(x, v)

        x = self.upsample(x)
        x = self.up2(torch.cat([x, x2], dim = 1), v)
        x = self.upsample(x)
        x = self.up1(torch.cat([x, x1], dim = 1), v)
        x = self.out(x)

        return x