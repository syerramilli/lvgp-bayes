import torch

def translate_and_rotate(Z):
    Zcen = Z-Z[...,[0],:]
    theta = torch.arctan(Zcen[...,[1],[1]]/Zcen[...,[1],[0]])
    c,s = theta.cos(),theta.sin()
    R = torch.cat([
        torch.stack([c,-s],dim=-1),
        torch.stack([s,c],dim=-1)
    ],dim=-2)
    return torch.matmul(Zcen,R)