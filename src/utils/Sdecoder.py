import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common import normalize_3d_coordinate

class nerf_pos_embed(nn.Module):
    def __init__(self, multires):
        super().__init__()
        self.max_freq_log2 = multires - 1
        self.periodic_fns = [torch.sin, torch.cos]
        self.num_freqs = multires
        self.max_freq = self.max_freq_log2
        self.N_freqs = self.num_freqs

    def forward(self,x):
        x = x.squeeze(0)
        freq_bands = torch.linspace(2. ** 0, 2. ** self.max_freq, step = self.N_freqs)
        output = []
        output.append(x)
        for freq in freq_bands:
            for p_fn in self.periodic_fns:
                output.append(p_fn(x * freq))
        return torch.cat(output, dim=1)        
    
class DenseLayer(nn.Linear):
    """
    Dense layer with activation.
    
    Args:
        in_dim (int): input dimension.
        out_dim (int): output dimension.
        activation (str, optional): activation function. Defaults to "relu".

    """
    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu", *args, **kwargs) -> None:
        self.activation = activation
        super().__init__(in_dim, out_dim, *args, **kwargs)

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight, gain = nn.init.calculate_gain(self.activation))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class MLP(nn.Module):
    """
    Decoder MLP.
    input: position & feature grids
    output: color & density grids ?
    Args:
        dim (int): input dimension.
        c_dim (int): feature dimension.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        bound (tensor, 3*2): the scene bound.
    """
    def __init__(self, dim = 3, c_dim = 128, hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', color=False, skips=[2], bound = torch.tensor([[-1, 1], [-1, 1], [-1, 1]])):
        super().__init__()
        self.bound = bound
        self.c_dim = c_dim
        self.n_blocks = n_blocks
        self.color = color
        self.skips = skips

        self.fc_c = nn.ModuleList([nn.Linear(c_dim, hidden_size) for i in range(n_blocks)]) # 5个全连接层
        multires = 10
        self.embedder = nerf_pos_embed(multires)    # 位置编码器
        embedding_size = multires * 6 + 3

        self.pts_linears = nn.ModuleList([DenseLayer(embedding_size, hidden_size, activation="relu")] +[DenseLayer(hidden_size, hidden_size, activation="relu") if i not in self.skips else DenseLayer(hidden_size + embedding_size, hidden_size, activation="relu") for i in range(n_blocks-1)])

        self.output_linear = DenseLayer(hidden_size, 4 if self.color else 1, activation="linear")

        self.actvn = lambda x: F.leaky_relu(x, 0.2) if leaky else F.relu

        self.sample_mode = sample_mode

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), self.bound)
        p_nor = p_nor.unsqueeze(0)
        vgrid = p_nor[:, :, None, None].float()
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c

    def forward(self, p, c_grid = None):
        c = self.sample_grid_feature(p, c_grid).transpose(1, 2).squeeze(0)  # 原本c_grid是一个字典，这里取消了分层结构
        p = p.float()

        embedded_pts = self.embedder(p)
        h = embedded_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            h = h + self.fc_c[i](c)
            if i in self.skips:
                h = torch.cat([embedded_pts, h], -1)
        out = self.output_linear(h)
        if not self.color:
            out = out.squeeze(-1)
        return out

class SNeEncoder(nn.Module):
    """
    Neural Implicit Scalable Encoding.
    
    Args:
        dim (int): input dimension.
        c_dim (int): feature dimension.
        n_blocks (int): number of layers.
        leaky (bool): whether to use leaky ReLUs.
        sample_mode (str): sampling feature strategy, bilinear|nearest.
        color (bool): whether or not to output color.
        skips (list): list of layers to have skip connections.
        bound (tensor, 3*2): the scene bound.

    """
    def __init__(self, slam):
        super().__init__()
        self.dim = slam.dim
        self.c_dim = slam.c_dim
        self.bounds = slam.bounds
        self.hidden_size = 32

        self.decoder = MLP(dim=self.dim, c_dim=self.c_dim, hidden_size=self.hidden_size, bound=self.bounds)
        
    def forward(self, p, c_grid):
        device = f'cuda:{p.get_device()}'
        occ = self.decoder(p, c_grid).squeeze(0)
        raw = torch.zeros(occ.shape[0], 4).to(device).float()
        raw[:, -1] = occ
        return raw
