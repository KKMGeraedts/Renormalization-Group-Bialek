import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F 

class RBM(nn.Module):

    def __init__(self, n_visible, n_hidden, CD_depth):
        super(RBM, self).__init__()
        # User input
        self.CD_depth = CD_depth

        # Model Parameters
        self.W = nn.Parameter(torch.randn(n_visible, n_hidden)) * 1e-2
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

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
        p_h = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias)
        return self.sample_from_p(p_h), p_h

    def h_to_v(self, h):
        p_v = torch.sigmoid(torch.matmul(h, self.W.t()) + self.v_bias)
        return self.sample_from_p(p_v), p_v

    def forward(self, v):
        vk = v
        for _ in range(self.CD_depth):
            hk, _ = self.v_to_h(vk)
            vk, _ = self.h_to_v(hk)
        return vk

    def CD_labels(self, v, k):
        # Perform k CD steps

        # Return labels
        pass

if __name__ == "__main__":
    # Initialize
    epochs = 10
    n_visible = v_data.shape[1]
    n_hidden = 500
    lr = 0.01
    CD_depth = 5
    batch_size = 50

    # Create RBM layer
    rbm = RBM(n_visible, n_hidden, CD_depth)

    # Optimizer
    optimizer = torch.optim.SGD(rbm.parameters(), lr=lr)

    # Things to keep track of
    loss_ = []

    for epoch in range(epochs):
        # Randomly permute the dataset
        v_data = v_data[torch.randperm(v_data.size()[0])]
        print(type(v_data))

        for i in range(0, len(v_data), batch_size):
            # Get batch
            v_batch = v_data[i:i+batch_size]

            # Forward
            h_batch = rbm.v_to_h(v_batch)
            v_forward = rbm(v_batch)
            h_forward = rbm.v_to_h(v_forward)

            # Compute loss
            loss = v_batch.mm(h_batch.T).mean() - v_forward.mm(h_forward.T).mean()
            loss_.append(loss)
            
            # Backward
            loss.backward()

            # Update weights
            optimizer.zero_grad()
            optimizer.step()

        # Print loss
        if (epoch+1) % 10 == 0:
            print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, 100, loss.item()))
