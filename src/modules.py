import torch
import torch.nn as nn
import torch.distributions as D

# --------------------
# Helper functions
# --------------------
def convolve_circular(a, b):
    '''
    a: vector
    b: kernel (must have odd length!)
    Naive implementation of circular convolution. The middle entry of b corresponds to the coefficient of z^0:
    b[0] b[1] b[2] b[3] b[4]
    z^-2 z^-1 z^0  z^1  z^2
    '''
    # Reshape to 2D array with each row a datapoint.
    a_orig_shape = a.shape
    a = a.reshape(-1, a.shape[-1])
    len_a = a.size()[1]
    len_b = b.size()[0]
    # print("len_a", len_a)
    # print(a.dtype)
    # print(b.dtype)
    result = torch.zeros(a.shape)
    for i in range(0, len_a):
        for j in range(0, len_b):
            result[:, i] += b[-j-1] * a[:, (i + (j - len_b//2)) % len_a]
    return result.reshape(a_orig_shape)

# --------------------
# Model component layers
# --------------------
class LazyWavelet(nn.Module):
    '''
    This layer corresponds to the downsampling step of a single-step wavelet transform.
    Input to forward: 1D tensor of even length.
    Output of forward: Two stacked 1D tensors of the form [[even x componenents], [odd x components]]
    See https://uk.mathworks.com/help/wavelet/ug/lifting-method-for-constructing-wavelets.html
    for notation.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_evens = x[0::2]
        x_odds = x[1::2]
        log_det = 0
        return torch.stack([x_evens, x_odds]), log_det

    def inverse(self, z):
        x_evens = z[0,:]
        x_odds = z[1,:]
        x = torch.reshape(torch.stack([x_evens, x_odds], axis=1), shape=[-1])  # interleave evens and odds
        log_det = 0
        return x, log_det

class Lifting(nn.Module):
    '''
    This layer corresponds to two elementary matrices of the polyphase matrix of a single-step wavelet transform.
    Input to forward: Two stacked 1D tensors of the form [[lowpass wavelet coefficients], [highpass wavelet coefficients]],
        i.e. the output of LazyWavelet or another Lifting layer.
    Output of forward: Two stacked 1D tensors of the form [[lowpass wavelet coefficients], [highpass wavelet coefficients]].
    See https://uk.mathworks.com/help/wavelet/ug/lifting-method-for-constructing-wavelets.html
    for notation.
    '''
    def __init__(self,
                # 2 coefficients are sufficient for fully general wavelet transforms,
                # given enough lifting steps. Use 3, because convolve_circular expects
                # an odd number.
                n_lifting_coeffs=3,
                P_coeff=torch.rand((3,)),
                U_coeff=torch.rand((3,))):
        super().__init__()
        self.n_lifting_coeffs = n_lifting_coeffs
        self.P_coeff = nn.Parameter(P_coeff)  # P: predict (primal lifting)
        self.U_coeff = nn.Parameter(U_coeff)  # U: update (dual lifting)

    def forward(self, x):
        x_evens = x[0,:]
        x_odds = x[1,:]
        evens_conv_P = convolve_circular(x_evens, self.P_coeff)
        detail = x_odds - evens_conv_P
        detail_conv_U = convolve_circular(detail, self.U_coeff)
        average = x_evens + detail_conv_U
        log_det = 0
        return torch.stack([average, detail]), log_det

    def inverse(self, z):
        average = z[0,:]
        detail = z[1,:]
        detail_conv_U = convolve_circular(detail, self.U_coeff)
        x_evens = average - detail_conv_U
        evens_conv_P = convolve_circular(x_evens, self.P_coeff)
        x_odds = evens_conv_P + detail
        x = torch.stack([x_evens, x_odds])
        log_det = 0
        return x, log_det

# --------------------
# Container layers
# --------------------
class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def __init__(self, *args, **kwargs):
        self.checkpoint_grads = kwargs.pop('checkpoint_grads', None)
        super().__init__(*args, **kwargs)

    def forward(self, x):
        sum_log_dets = 0.
        for module in self:
            x, log_det = module(x)
            sum_log_dets = sum_log_dets + log_det
        return x, sum_log_dets

    def inverse(self, z):
        sum_log_dets = 0.
        for module in reversed(self):
            z, log_det = module.inverse(z)
            sum_log_dets = sum_log_dets + log_det
        return z, sum_log_dets

class WaveletStep(FlowSequential):
    '''
    One step in a wavelet transform.
    '''
    def __init__(self, n_lifting_steps):
        liftings = [Lifting() for _ in range(0, n_lifting_steps)]
        lazy_wavelet = LazyWavelet()
        super().__init__(lazy_wavelet, *liftings)

# TODO
class Wavelet(nn.Module):
    pass

# --------------------
# Model
# --------------------
class WaveletNet(nn.Module):
    '''
           _.====.._
         ,:._       ~-_
             `\        ~-_
               | _  _  |  `.
             ,/ /_]/ | |    ~-_
    -..__..-''  \_ \_\ `_      ~~--..__...----...
    '''
    def __init__(self):
        super().__init__()
        self.wavelet_step = WaveletStep(n_lifting_steps=2)
        self.base_dist = D.Normal(0., 1.)

    def forward(self, x):
        return self.wavelet_step.forward(x)

    def inverse(self, z):
        self.wavelet_step.inverse(z)

    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_prob = sum(self.base_dist.log_prob(z)) + log_det
        return log_prob
