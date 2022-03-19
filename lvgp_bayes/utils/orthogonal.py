import torch
import math

def _vec_to_lowtri(x):
    '''
    
    '''
    batch_shape = torch.Size([]) if x.ndim == 1 else torch.Size([x.shape[0]])
    m = x.shape[-1]
    n = int(math.sqrt(0.25 + 2. * m) + 0.5)
    z = torch.zeros(*batch_shape,n,n).to(x)
    idx_start=0
    for i in range(n-1):
        num_params = n-1-i
        z[...,(i+1):,i] = x[...,idx_start:(idx_start+num_params)]
        idx_start += num_params
    
    return z

def eye_like(X):
    Id = torch.zeros(*X.shape[:-2],X.shape[-2],X.shape[-2]).to(X)
    Id.diagonal(dim1=-2,dim2=-1).fill_(1.)
    return Id

# def construct_orthogonal(first_block_vec,second_block):
#     V = torch.cat([_vec_to_lowtri(first_block_vec),second_block],dim=-2)
#     #A = V.tril(diagonal=-1)
#     tau = 2. / (1. + (V * V).sum(dim=-2))
#     Q = torch.linalg.householder_product(V, tau)
    
#     return Q

# def construct_orthogonal(first_block_vec,second_block):
#     V = torch.cat([_vec_to_lowtri(first_block_vec),second_block],dim=-2)
#     m,n = V.shape[-2],V.shape[-1]
#     Id = eye_like(V)

#     Q = eye_like(V)

#     for i in range(n):
#         vi = torch.cat([torch.ones(*V.shape[:-2],1,1).to(V),V[...,(i+1):,[i]]],dim=-2)
#         tau_i = 2./(vi**2).sum(dim=[-2,-1]).view(*V.shape[:-2],1,1)
        
#         H_i = torch.cat([
#             torch.cat([
#                 Id[...,:(i),:(i)],torch.zeros(*V.shape[:-2],i,m-i).to(V)
#             ],dim=-1),
#             torch.cat([
#                 torch.zeros(*V.shape[:-2],m-i,i).to(V),
#                 torch.add(Id[...,i:,i:],tau_i*vi@vi.transpose(-2,-1),alpha=-1)
#             ],dim=-1)
#         ],dim=-2)
    
#         # update Q
#         Q = Q@H_i
    
#     return Q[...,:n]


def construct_orthogonal(raw_skew_vec,raw_weight):
    '''
    Construct orthogonal matrix via Cayley transform of a skew-symmetric matrix
    '''
    batch_size = raw_weight.shape[:-2]
    diff_dims,out_dim = raw_weight.shape[-2:]
    
    # construct skew symmetric matrix
    B = _vec_to_lowtri(raw_skew_vec)
    B = B - B.transpose(-2,-1)
    
    # main skew symmetric matrix
    A = torch.cat([
        torch.cat([
           B,-raw_weight.transpose(-2,-1)
        ],dim=-1),
        torch.cat([
            raw_weight,
            torch.zeros(*batch_size,diff_dims,diff_dims).to(raw_weight)
        ],dim=-1)
    ],dim=-2)
    
    # return torch.matrix_exp(A)[...,:out_dim]
    # Computes the Cayley retraction (I+A/2)(I-A/2)^{-1}
    Id = torch.eye(A.shape[-2], dtype=A.dtype, device=A.device)
    Q = torch.linalg.solve(torch.add(Id, A, alpha=-0.5), torch.add(Id, A, alpha=0.5))
    
    return Q[...,:out_dim]