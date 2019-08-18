import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from utils import get_polyphase_matrix_product, get_polyphase_identity_matrix, convolve_circular

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

    WORKS ONLY WITH len(x) = 2^x, integer x.
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x_evens = x[:,0::2]
        x_odds = x[:,1::2]
        log_det = 0
        return torch.stack([x_evens, x_odds], dim=1), log_det

    def inverse(self, z):
        x_evens = z[:,0,:]
        x_odds = z[:,1,:]
        x = torch.reshape(torch.stack([x_evens, x_odds], axis=2), shape=[-1, 2*z.shape[-1]])  # interleave evens and odds
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
                p_lifting_coeffs=3,
                u_lifting_coeffs=3):
        super().__init__()
        self.P_coeff = nn.Parameter(torch.zeros((p_lifting_coeffs,)))  # P: predict (primal lifting)
        self.U_coeff = nn.Parameter(torch.zeros((u_lifting_coeffs,)))  # U: update (dual lifting)

    def forward(self, x):
        x_evens = x[:,0,:]
        x_odds = x[:,1,:]
        evens_conv_P = convolve_circular(x_evens, self.P_coeff)
        detail = x_odds - evens_conv_P
        detail_conv_U = convolve_circular(detail, self.U_coeff)
        average = x_evens + detail_conv_U
        log_det = 0
        return torch.stack([average, detail], dim=1), log_det

    def inverse(self, z):
        average = z[:,0,:]
        detail = z[:,1,:]
        detail_conv_U = convolve_circular(detail, self.U_coeff)
        x_evens = average - detail_conv_U
        evens_conv_P = convolve_circular(x_evens, self.P_coeff)
        x_odds = evens_conv_P + detail
        x = torch.stack([x_evens, x_odds], dim=1)
        log_det = 0
        return x, log_det

    def get_polyphase_matrix(self):
        return get_polyphase_matrix_product(
            [torch.tensor([1.]), self.U_coeff.clone().detach(),
             torch.tensor([0.]), torch.tensor([1.])],
            [torch.tensor([1.]), torch.tensor([0.]),
             -self.P_coeff.clone().detach(), torch.tensor([1.])],
        )

class FlattenLatents(nn.Module):
    def __init__(self, n_wavelet_steps):
        super().__init__()
        self.n_wavelet_steps = n_wavelet_steps

    def forward(self, x):
        return torch.cat(x, dim=1), 0

    def inverse(self, z):
        zs = []
        pos = 0
        for i in range(0, self.n_wavelet_steps):
            n_coeffs = (z.shape[1]) // (2**(i+1))
            zs.append(z[:, pos:pos+n_coeffs])
            pos += n_coeffs
        zs.append(z[:, pos:z.shape[1]])
        return zs, 0

class LeakyReLU(nn.Module):
    def __init__(self, alpha=.5):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return (torch.where(x >= 0, x, self.alpha*x), -self._inverse_log_det_jacobian(x))

    def inverse(self, z):
        return (torch.where(z >= 0, z, (1./self.alpha)*z), self._inverse_log_det_jacobian(z))

    def _inverse_log_det_jacobian(self, z):
        I = torch.ones(z.shape, device=z.device)
        J_inv = torch.where(z >= 0, I, 1.0 / self.alpha * I)
        # abs is actually redundant here, since this det Jacobian is > 0
        log_abs_det_J_inv = torch.log(torch.abs(J_inv))
        return torch.sum(log_abs_det_J_inv, dim=1)

# --------------------
# Container layers
# --------------------
class FlowSequential(nn.Sequential):
    """ Container for layers of a normalizing flow """
    def __init__(self, *args, **kwargs):
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
    dims: array of shape (n_lifting_steps, 2)
          dims[k, 0] gives p_lifting_coeffs for the kth lifting step.
          dims[k, 1] gives u_lifting_coeffs for the kth lifting step.
          Default of 2 lifting steps with 3 coefficients per filter.
    '''
    def __init__(self, dims=[[3,3], [3,3]]):
        self.dims = dims
        self.liftings = [Lifting(p_lifting_coeffs, u_lifting_coeffs) for (p_lifting_coeffs, u_lifting_coeffs) in dims]
        lazy_wavelet = LazyWavelet()
        super().__init__(lazy_wavelet, *self.liftings)

    '''
    Returns the polyphase matrix M = [a b
                                      c d]
    as [a, b, c, d], where M is the product of the polyphase matrices of the lifting steps.
    '''
    def get_polyphase_matrix(self):
        M = get_polyphase_identity_matrix()
        for lifting in self.liftings:
            M = get_polyphase_matrix_product(lifting.get_polyphase_matrix(), M)
        return M

class Wavelet(nn.Module):
    '''
    Multi-step wavelet transform.
    dims: array of shape (n_wavelet_steps, n_lifting_steps, 2)
          dims[j, k, 0] gives p_lifting_coeffs for the kth lifting step in the jth wavelet step.
          dims[j, k, 1] gives u_lifting_coeffs for the kth lifting step in the jth wavelet step.
          Default of 2 wavelet steps with 2 lifting steps with 3 coefficients per filter.
    share_parameters: If True, share Parameters across all WaveletStep layers. The Wavelet then corresponds
        to a valid wavelet transform.
    '''
    def __init__(self, dims=[[[3,3], [3,3]], [[3,3], [3,3]]], share_parameters=False):
        super().__init__()
        if share_parameters:
            self._assert_valid_dims_for_shared_parameters(dims)
            wavelet_step = WaveletStep(dims[0])
            self.wavelet_steps = nn.ModuleList([wavelet_step for _ in dims])
        else:
            self.wavelet_steps = nn.ModuleList([WaveletStep(wavelet_step_dims) for wavelet_step_dims in dims])

    def _assert_valid_dims_for_shared_parameters(self, dims):
        if not all(x == dims[0] for x in dims):
            raise Exception("When sharing parameters, all entries in dims must be equal!")

    def forward(self, x):
        zs = []
        lowpass_coeffs = x
        sum_log_dets = 0.
        for wavelet_step in self.wavelet_steps:
            coeffs, log_det = wavelet_step(lowpass_coeffs)
            zs.append(coeffs[:,0])
            lowpass_coeffs = coeffs[:,1]
            sum_log_dets += log_det
        zs.append(lowpass_coeffs)
        return zs, sum_log_dets

    def inverse(self, zs):
        sum_log_dets = 0.
        zs_idx = -1
        lowpass_coeffs = zs[zs_idx]
        for wavelet_step in reversed(self.wavelet_steps):
            zs_idx -= 1
            z = zs[zs_idx]
            lowpass_coeffs, log_det = wavelet_step.inverse(torch.stack([z, lowpass_coeffs], dim=1))
            sum_log_dets += log_det
        return lowpass_coeffs, sum_log_dets

class Bijector(FlowSequential):
    def __init__(self,*args, base_dist = D.Normal(0, 1), **kwargs):
        print(args, base_dist, kwargs)
        super().__init__(*args, **kwargs)
        self.base_dist = base_dist

    def sample(self, n_samples, sample_length):
        z = self.base_dist.sample((n_samples, sample_length))
        with torch.no_grad():
            x, _ = self.inverse(z)
        return x.numpy()

    def log_prob(self, x):
        z, log_det = self.forward(x)
        log_prob = self.base_dist.log_prob(z).sum([1]) + log_det
        return log_prob

# --------------------
# Model
# --------------------
class WaveletNet(Bijector):
    '''
           _.====.._
         ,:._       ~-_
             `\        ~-_
               | _  _  |  `.
             ,/ /_]/ | |    ~-_
    -..__..-''  \_ \_\ `_      ~~--..__...----...
    '''
    def __init__(self, dims=[[[3,3], [3,3]], [[3,3], [3,3]]], share_parameters=False):
        super().__init__(Wavelet(dims, share_parameters), FlattenLatents(len(dims)))
