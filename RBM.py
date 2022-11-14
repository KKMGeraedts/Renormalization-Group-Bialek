import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

class RBM(nn.Module):

    def __init__(self, n_visible, n_hidden, lr, CD_depth=5):
        super(RBM, self).__init__()
        # User input
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.CD_depth = 5

        # Model Parameters
        initial_weight_variance = 1e-2
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden)) * initial_weight_variance
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def sample_from_p(p):
        """
        Sample a binary value from the probability distribution p. p is a 1d array with each value i corresponding
        to the probability that variable i is equal to 1. Naturally 1-p is the probablity of 0 then.

        Parameters:
            p - probability distribution

        Return:
            sample - the binary sample drawn from p
        """
        uniform_random = torch.rand(p.size)
        sample = F.relu(torch.sign(p  - uniform_random))
        return sample

    def v_to_h(self, v):
        h = F.linear(v, self.W, self.h_bias)
        p_h = F.sigmoid(h)
        h = self.sample_from_p(p_h)
        return h

    def h_to_v(self, h):
        v = F.linear(h, self.W.T, self.v_bias)
        p_v = F.sigmoid(v)
        v = self.sample_from_p(p_v)
        return v

    def forward(self, v):
        for _ in range(self.CD_depth):
            h = self.v_to_h(v)
            v = self.h_to_v(h)
        return v

if __name__ == "__main__":
    # Initialize
    epochs = 10
    n_visible = 400
    n_hidden = 200
    lr = 0.01
    CD_depth = 5

    # Create RBM layer
    rbm = RBM(n_visible, n_hidden, lr, CD_depth)

    # Read data
    v_data = np.random.randn(400, 10000)
    v = v_data

    # Things to keep track of
    loss_ = []

    for _ in range(epochs):
        # Pass data through network
        h_data = rbm.v_to_h(v)
        v_forward = rbm(v)
        h_forward = rbm.v_to_h(v_forward)

        # Compute loss
        loss = v_data.mm(h_data.T).mean() - v_forward.mm(h_forward.T).mean()
        loss_.append(loss)
        
        # Update weights
        loss.backward()
        print(v_data.mm(h_data.T).mean().shape)



