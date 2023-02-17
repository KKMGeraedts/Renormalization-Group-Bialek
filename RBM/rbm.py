import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision.datasets as datasets
from torch.optim import Optimizer
from data_cleaning import *
import time

class RBMOptimizer(Optimizer):
    def __init__(self, rbm, learning_rate=0.01, weight_decay=0, momentum=0):
        defaults = dict(learning_rate=learning_rate, weight_decay=weight_decay, momentum=momentum)
        super().__init__(rbm.parameters(), defaults)
        self.rbm = rbm

    def step(self, v_bias_update, h_bias_update, vh_weight_update):
        for group in self.param_groups:
            # Learning params
            learning_rate = group["learning_rate"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]

            # Update weights
            self.rbm.v_bias += learning_rate * v_bias_update
            self.rbm.h_bias += learning_rate * h_bias_update
            self.rbm.W += learning_rate * vh_weight_update #- weight_decay * torch.sign(self.rbm.W)

class RBM(nn.Module):

    def __init__(self, n_visible, n_hidden):
        super(RBM, self).__init__()
        # NOTE: Set requires_grad=False because the weights are updated using CD not with backpropagation
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden) * 1e-2, requires_grad=False)
        self.v_bias = nn.Parameter(torch.zeros(n_visible), requires_grad=False)
        self.h_bias = nn.Parameter(torch.zeros(n_hidden), requires_grad=False)

    def sample_from_p(self, p):
        """
        Sample a binary value from the probability distribution p. p is a 1d array with each value i corresponding
        to the probability that variable i is equal to 1. Naturally 1-p is the probablity of 0 then.

        Parameters:
            p - probability distribution

        Return:
            sample - the binary sample drawn from p
        """
        # uniform_random = torch.rand(p.size)
        # sample = F.relu(torch.sign(p  - uniform_random))
        # return sample
        return torch.bernoulli(p)

    def v_to_h(self, v):
        p_h = torch.sigmoid(v @ self.W + self.h_bias)
        return self.sample_from_p(p_h), p_h

    def h_to_v(self, h):
        p_v = torch.sigmoid(h @ self.W.t() + self.v_bias)
        return self.sample_from_p(p_v), p_v

    def forward(self, v_probs, h_probs):
        # Return prediction (v_i * h_j)

        # NOTE: The prediction is the joint probability of the two layers firing.
        return (v_probs.T @ h_probs)


    def create_CD_reconstructions(self, h_states, h_probs, k):
        # NOTE: When creating the labels we do not want torch to update the gradients.
        # NOTE: For the learning I am using the binary values. Check whether learning is quicker with binary values or probabilities instead?
        with torch.no_grad():
            hk_states = h_states
            hk_probs = h_probs
            len_data_batch = h_states.size()[0]

            # Perform k CD steps
            for _ in range(k-1):
                vk_states, vk_probs = self.h_to_v(hk_states)
                hk_states, hk_probs = self.v_to_h(vk_probs)

            # Last update is slightly different
            vk_states, vk_probs = self.h_to_v(hk_probs) #NOTE: Here we use the hidden probabilities instead (This should avoid unnecessary sampling noise)
            kh_states, hk_probs = self.v_to_h(vk_probs)

            # CD visible and hidden state reconstructions
            h_reconstruction = hk_probs
            v_reconstruction = vk_probs
            vh_reconstruction = (vk_probs.T @ hk_probs)# / len_data_batch

            # Return CD reconstructions
            return vh_reconstruction, v_reconstruction, h_reconstruction

    def train(self, v_data, epochs=5000, batch_size=0, CD_depth=5, learning_rate=0.1, weight_decay=0.1, momentum=0, verbose=True, optimizer=None):

        # If no batch size is given the us the entire dataset
        if batch_size == 0:
            batch_size = len(v_data)

        # Optimizer
        if optimizer == None:
            optimizer = RBMOptimizer(self, learning_rate=learning_rate, weight_decay=weight_decay, momentum=momentum)

        for epoch in range(epochs):
            # Initialize errors (per epoch) at zero
            v_recon_error = torch.tensor(0, dtype=float)
            h_recon_error = torch.tensor(0, dtype=float)
            vh_recon_error = torch.tensor(0, dtype=float)

            v_probs_batch = v_data

            # Put data in a torch.DataLoader
            # v_dataloader = torch.utils.data.DataLoader(v_data, batch_size=batch_size, shuffle=True, num_workers=0)
            
            # Start batch training
            # for v_probs_batch in v_dataloader:
            # NOTE: Adding this seems to slow down the computation. Probably because computation is not sent to the GPU yet. 
            # NOTE: For now that means we just train by computing gradient over the entire d    taset. I.e. no SGD.
            
            # Forward statistics
            h_states_batch, h_probs_batch = self.v_to_h(v_probs_batch)
            vh_batch = self(v_probs_batch, h_probs_batch) #NOTE: Joint-probability distribution P(v, h)   

            # Reconstruction statistics (Computed using Contrastive Divergence)
            vh_recon, v_probs_recon, h_probs_recon = self.create_CD_reconstructions(h_states_batch, h_probs_batch, CD_depth)

            # Compute weight updates
            v_bias_update = (v_probs_batch - v_probs_recon).mean(axis=0)
            h_bias_update = (h_states_batch - h_probs_recon).mean(axis=0)
            vh_weight_update = (vh_batch - vh_recon) / len(v_probs_batch)

            # Update weights    
            optimizer.step(v_bias_update, h_bias_update, vh_weight_update)
            
            # Compute loss statistics
            v_recon_error += torch.sum((v_probs_batch - v_probs_recon) ** 2, dim=(0,1))
            h_recon_error += torch.sum((h_states_batch - h_probs_recon) ** 2, dim=(0,1))
            vh_recon_error += torch.sum((vh_batch - vh_recon) ** 2, dim=(0,1))

            # Store losses together
            loss = [v_recon_error, h_recon_error, vh_recon_error]

            # Print loss
            if verbose == True:
                if (epoch+1) % 1 == 0:
                    print ('Epoch [{}/{}], Reconstructions errors: {}'.format(epoch+1, epochs, loss))
            else:
                if epoch+1 == epochs:
                    print ('Epoch [{}/{}], Reconstructions errors: {}'.format(epoch+1, epochs, loss))

def change_binary_data_to_probabilities(v_data):
    """
    When training RBMs one needs to train the model on probabilities and not the binary data themselves.
    So here we compute the average probability of each visible spin being up or down and change the binary values
    in the dataset to now contain those probabilities.

    Parameters:
        v_data - Dataset of binary (0, 1) values

    Return:
        v_data_probs - Same dataset not containing (p(0), p(1)) instead.
    """
    # Compute average 
    v_prob_spin_up = v_data.mean(dim=0)
    v_prob_spin_down = 1 - v_prob_spin_up

    # Create array of same shape as dataset. We can then mask this.
    v_prob_spin_up = v_prob_spin_up.repeat(len(v_data), 1)
    v_prob_spin_down = v_prob_spin_down.repeat(len(v_data), 1)

    #NOTE: Use the dataset as a mask te create seperate ndarrays with P(v=1) and P(v=0)
    # Masks
    mask_spin_up = v_data.bool()
    mask_spin_down = (1 - v_data).bool()

    # New ndarrays
    v_data_probs_spin_up = torch.where(mask_spin_up, v_prob_spin_up, 0)
    v_data_probs_spin_down = torch.where(mask_spin_down, v_prob_spin_down, 0)

    # Now add them together to get dataset in terms of probabilities
    v_data_probs = v_data_probs_spin_down + v_data_probs_spin_up
    return v_data_probs

if __name__ == "__main__":
    # Load dataset
    f = "data/2d_Ising_metropolis_L=16,T=3.npy"
    v_data = read_input(f)
    v_data[v_data == - 1] = 0 # Transform data to 0s and 1s

    # Use torch for computation
    v_data = torch.tensor(v_data).float()
    
    # Transform binary data to probabilities
    v_data_probs = change_binary_data_to_probabilities(v_data)

    #### Initialize parameters ###
    # Layer params
    n_visible = v_data_probs.shape[1]
    n_hidden = n_visible // 4

    # Optimizer params
    learning_rate = 0.1
    weight_decay = 0.1
    momentum = 0

    # Learning params
    epochs = 1000
    batch_size = 3200

    # Create RBM
    rbm = RBM(n_visible, n_hidden)

    # Train model
    rbm.train(v_data_probs, epochs=epochs)

