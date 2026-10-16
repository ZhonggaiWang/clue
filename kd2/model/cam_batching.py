"""Memory-bounded no-grad CAM evaluation; training loss batch remains intact."""
import torch
from utils.camutils import multi_scale_cam2
@torch.no_grad()
def memory_bounded_cams(model, images, scales, chunk_size=4):
    if chunk_size < 1: raise ValueError('chunk_size must be positive')
    chunks = [multi_scale_cam2(model, x, scales) for x in images.split(chunk_size, dim=0)]
    return tuple(torch.cat([pair[i] for pair in chunks], dim=0) for i in [0,1])
