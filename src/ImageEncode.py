import numpy as np
from scipy.stats import unitary_group
from skimage.measure import block_reduce
from opt_einsum import contract
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
import torch
from qutip import *
import time
import sys
import matplotlib.pyplot as plt


""" 
    Steps in encoding an image into quantum states

We want n qubits to hold 2**n states. Each state will represent the "value" of 
a dimension of a pixel of an image. To this end:

1. Normalize the signal. 

2. Complexify signal so that "value" is the probability axis in the bloch sphere


"""

def normalize(values: np.array):
    '''
    Normalizes inputs 
    Args:
        values: arrays of pixel "values" in a single dimension.
            i.e. for green, can contain [ [0, 1], [255], [2]]
    Output:
        flattened normalized values of pixels
    '''
    values = values.flatten()
    norm: float = np.linalg.norm(values)
    normalized_image_values: np.array = values/norm
    return normalized_image_values
    
def complexify(values: np.array):
    """
    Takes values as complexes

    Args:
        values (np.array): normalized amplitudes
    """
    return values + 0j
    
def generate_training(values: np.array, n_train: int, scale: float, seed=None):
    np.random.seed(seed)
    n = values.size
    noise = np.random.randn(n_train,n)+1j*np.random.randn(n_train,n) 
    print(noise.shape)
    print(values.shape)
    states = (noise*scale) + values
    states/=np.tile(np.linalg.norm(states, axis=1).reshape(1,n_train), (n,1)).T
    return states
        
    
    
def bloch_xyz(inputs):
    # obtain bloch sphere representation vector
    rho = contract('mi,mj->mij', inputs, inputs.conj())
    sigmas = [qutip.sigmax().full(), qutip.sigmay().full(), qutip.sigmaz().full()]
    pos = [np.real(contract('mii->m', contract('mij,jk->mik', rho, x))) for x in sigmas]
    return pos
    
def bloch_plot(x, y, z):
    fig, axs = plt.subplots(1,1, subplot_kw={'projection': '3d'})
    cc = 0
    b0 = qutip.Bloch(fig=fig, axes=axs[0,cc])
    b0.clear()
    b0.add_points([x, y, z])
    b0.point_color = ['r']*Ndata
    b0.point_style = 'm'
    b0.point_size = 8*np.ones(Ndata)
    b0.render()
    b1 = qutip.Bloch(fig=fig, axes=axs[1,cc])
    b1.clear()
    b1.add_points([xs1, ys1, zs1])
    b1.point_color = ['b']*Ndata
    b1.point_style = 'm'
    b1.point_size = 8*np.ones(Ndata)
    b1.render()
    b2 = qutip.Bloch(fig=fig, axes=axs[2,cc])
    b2.clear()
    b2.add_points([xs2, ys2, zs2])
    b2.point_color = ['forestgreen']*Ndata
    b2.point_style = 'm'
    b2.point_size = 8*np.ones(Ndata)
    b2.render()
    axs[0,cc].set_title(r'$t=%d$'%i, fontsize=20)
    cc += 1