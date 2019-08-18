import torch
import torch.nn.functional as F

# --------------------
# Helper functions
# --------------------

# Use this custom correct padding function because of
# https://github.com/pytorch/pytorch/issues/24504
def padding1d_circular(input, pad):
    left_pad = input[:, :, -pad[0]:]
    right_pad = input[:, :, 0:pad[1]]
    if pad[0] == 0:
        return torch.cat([input, right_pad], dim=2)
    else:
        return torch.cat([left_pad, input, right_pad], dim=2)


def convolve_circular(a, b):
    b = torch.flip(b, dims=[-1])
    b_len = b.shape[-1]
    return F.conv1d(
        padding1d_circular(a.unsqueeze(1), (b_len-1, 0)),
        b.unsqueeze(0).unsqueeze(0)
    ).squeeze(1)

def convolve_full(a, b):
    b = torch.flip(b, dims=[-1])
    b_len = b.shape[-1]
    return F.conv1d(
        F.pad(a.unsqueeze(1), (b_len-1, b_len-1)),
        b.unsqueeze(0).unsqueeze(0)
    ).squeeze(1)

def add_tensors_of_unequal_length(a, b):
    if a.shape[0] < b.shape[0]:
        a, b = b, a
    res = a.clone().detach()
    res[0:b.shape[0]] += b
    return res

def get_polyphase_matrix_product(M1, M2):
    def compute_entry(row, col):
        return add_tensors_of_unequal_length(
            convolve_full(M1[row*2].unsqueeze(0), M2[col]).squeeze(0),
            convolve_full(M1[row*2+1].unsqueeze(0), M2[col+2]).squeeze(0)
        )
    return [
        compute_entry(0, 0), compute_entry(0, 1),
        compute_entry(1, 0), compute_entry(1, 1)
    ]
    raise Exception("TODO")

def get_polyphase_identity_matrix():
    return [torch.tensor([1.]), torch.tensor([0.]),
            torch.tensor([0.]), torch.tensor([1.])]


'''
Returns a tuple (z^-1*h, z^-1*g), where h are the lowpass filter coefficients,
and g are the highpass filter coefficients.
'''
def get_wavelet_filters_from_polyphase_matrix(M):
    a, b, c, d = M[0], M[1], M[2], M[3]
    h = torch.zeros(a.shape[0]*2+1)
    g = torch.zeros(c.shape[0]*2+1)
    for i, coeff in enumerate(a):
        h[2*i+1] += coeff
    for i, coeff in enumerate(b):
        h[2*i] += coeff
    for i, coeff in enumerate(c):
        g[2*i+1] += coeff
    for i, coeff in enumerate(d):
        g[2*i] += coeff
    return (h, g)
