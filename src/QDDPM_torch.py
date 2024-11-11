import numpy as np
import ot
import tensorcircuit as tc
import scipy as sp
from scipy.stats import unitary_group
import torch
import torch.nn as nn
from torch.linalg import matrix_power
from opt_einsum import contract
from functools import partial
from itertools import combinations

# Set tensorcircuit to use PyTorch backend with complex64 dtype
K = tc.set_backend('pytorch')
tc.set_dtype('complex64')

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffusionModel(nn.Module):
    def __init__(self, n, T, Ndata):
        '''
        the diffusion quantum circuit model to scramble arbitrary set of states to Haar random states
        Args:
        n: number of qubits
        T: number of diffusion steps
        Ndata: number of samples in the dataset
        '''
        super().__init__()
        self.n = n
        self.T = T
        self.Ndata = Ndata
    
    def HaarSampleGeneration(self, Ndata, seed):
        '''
        generate random haar states,
        used as inputs in the t=T step for backward denoise
        Args:
        Ndata: number of samples in dataset
        '''
        np.random.seed(seed)
        states_T = unitary_group.rvs(dim=2**self.n, size=Ndata)[:,:,0]
        return torch.from_numpy(states_T).cfloat().to(device)
    
    def scrambleCircuit_t(self, t, input, phis, gs=None):
        '''
        obtain the state through diffusion step t
        Args:
        t: diffusion step
        input: the input quantum state
        phis: the single-qubit rotation angles in diffusion circuit
        gs: the angle of RZZ gates in diffusion circuit when n>=2
        '''
        c = tc.Circuit(self.n, inputs=input)
        for tt in range(t):
            for i in range(self.n):
                c.rz(i, theta=phis[3*self.n*tt+i])
                c.ry(i, theta=phis[3*self.n*tt+self.n+i])
                c.rz(i, theta=phis[3*self.n*tt+2*self.n+i])
            if self.n >= 2:
                for i, j in combinations(range(self.n), 2):
                    c.rzz(i, j, theta=gs[tt]/(2*np.sqrt(self.n)))
        return c.state().to(device)
    
    def set_diffusionData_t(self, t, inputs, diff_hs, seed):
        np.random.seed(seed)
        var_1 = diff_hs.repeat(3*self.n).to(device) #edited this
        phis = (torch.rand(self.Ndata, 3*self.n*t, device=device)*np.pi/4. - np.pi/8.) * var_1
        gs = (torch.rand(self.Ndata, t, device=device)*0.2 + 0.4 if self.n > 1 else None).to(device)
        diff_hs = diff_hs.to(device) #edited this
        gs = gs * diff_hs if gs is not None else None
        states = torch.zeros((self.Ndata, 2**self.n), dtype=torch.cfloat, device=device)
        for i in range(self.Ndata):
            states[i] = self.scrambleCircuit_t(t, inputs[i], phis[i], gs[i] if gs is not None else None)
        return states


def backCircuit(input, params, n_tot, L):
    '''
    the backward denoise parametric quantum circuits,
    designed following the hardware-efficient ansatz
    output is the state before measurements on ancillas
    '''
    c = tc.Circuit(n_tot, inputs=input)
    for l in range(L):
        for i in range(n_tot):
            c.rx(i, theta=params[2*n_tot*l+i])
            c.ry(i, theta=params[2*n_tot*l+n_tot+i])
        for i in range(n_tot//2):
            c.cz(2*i, 2*i+1)
        for i in range((n_tot-1)//2):
            c.cz(2*i+1, 2*i+2)
    return c.state().to(device)


class QDDPM(nn.Module):
    def __init__(self, n, na, T, L):
        '''
        the QDDPM model: backward process only work on CPU
        '''
        super().__init__()
        self.n = n
        self.na = na
        self.n_tot = n + na
        self.T = T
        self.L = L
        self.backCircuit_vmap = K.vmap(partial(backCircuit, n_tot=self.n_tot, L=L), vectorized_argnums=0)

    def set_diffusionSet(self, states_diff):
        self.states_diff = torch.from_numpy(states_diff).cfloat().to(device)

    def randomMeasure(self, inputs):
        m_probs = (torch.abs(inputs.reshape(inputs.shape[0], 2**self.na, 2**self.n))**2).sum(dim=2)
        m_res = torch.multinomial(m_probs, num_samples=1).squeeze()
        indices = 2**self.n * m_res.view(-1, 1) + torch.arange(2**self.n, device=device)
        post_state = torch.gather(inputs, 1, indices)
        norms = torch.sqrt(torch.sum(torch.abs(post_state)**2, axis=1)).unsqueeze(dim=1)
        return post_state / norms

    def backwardOutput_t(self, inputs, params): #most important thing to alter for gpu
        inputs = inputs.to(device)
        params = params.to(device)
        output_full = self.backCircuit_vmap(inputs, params)
        output_t = self.randomMeasure(output_full)
        output_t = output_t.to(device)
        return output_t
    
    def prepareInput_t(self, inputs_T, params_tot, t, Ndata):
        self.input_tplus1 = torch.zeros((Ndata, 2**self.n_tot), dtype=torch.cfloat, device=device)
        self.input_tplus1[:, :2**self.n] = inputs_T.to(device)
        params_tot = torch.from_numpy(params_tot).float().to(device)
        with torch.no_grad():
            for tt in range(self.T-1, t, -1):
                self.input_tplus1[:, :2**self.n] = self.backwardOutput_t(self.input_tplus1, params_tot[tt])
        return self.input_tplus1
    
    def backDataGeneration(self, inputs_T, params_tot, Ndata):
        states = torch.zeros((self.T+1, Ndata, 2**self.n_tot), dtype=torch.cfloat, device=device)
        states[-1, :, :2**self.n] = inputs_T.to(device)
        params_tot = torch.from_numpy(params_tot).float().to(device)
        with torch.no_grad():
            for tt in range(self.T-1, -1, -1):
                states[tt, :, :2**self.n] = self.backwardOutput_t(states[tt+1], params_tot[tt])
        return states


def naturalDistance(Set1, Set2):
    r11 = 1. - torch.mean(torch.abs(contract('mi,ni->mn', Set1.conj(), Set1))**2)
    r22 = 1. - torch.mean(torch.abs(contract('mi,ni->mn', Set2.conj(), Set2))**2)
    r12 = 1. - torch.mean(torch.abs(contract('mi,ni->mn', Set1.conj(), Set2))**2)
    return 2*r12 - r11 - r22


def WassDistance(Set1, Set2):
    D = 1. - torch.abs(Set1.conj() @ Set2.T)**2.
    emt = torch.empty(0, device=device)
    Wass_dis = ot.emd2(emt, emt, M=D.cpu())  # EMD function may need CPU
    return Wass_dis


def sinkhornDistance(Set1, Set2, reg=0.005, log=False):
    D = 1. - torch.abs(Set1.conj() @ Set2.T)**2.
    emt = torch.empty(0, device=device)
    sh_dis = ot.sinkhorn2(emt.cpu(), emt.cpu(), M=D.cpu(), reg=reg, method='sinkhorn_log' if log else None)
    return sh_dis
