from rbm import RBM
import numpy as np
import torch

def uniform_dist_test():
    # Testing the model on a simple dataset with 5 spins. P(v) is uniform.
    all_states = np.arange(2**5)
    all_states = [np.binary_repr(x, 5) for x in all_states]
    v_data_empty = np.empty(shape=(32, 5))

    for i, binary_string in enumerate(all_states):
        v_data_empty[i] = np.array([int(char) for char in binary_string], dtype=np.float16)

    all_states = torch.tensor(v_data_empty, dtype=torch.float)

    v_probs_spin_up = all_states.mean(dim=0)

    v_probs_spin_up = v_probs_spin_up.repeat(len(all_states), 1)
    all_states = all_states.bool()

    all_states_probs = torch.where(all_states, v_probs_spin_up, 0)
    all_states_probs[all_states_probs == 0.0] = 0.5 #NOTE: this will not work if the visible units have different P(silence)

    v_data_probs = np.repeat(all_states_probs, 100, axis=0)

    # Test params
    n_visible = 5
    n_hidden = 5
    batch_size = 320
    epochs = 5000
    CD_depth = 1
    learning_rate = 0.1

    # Create RBM
    rbm = RBM(n_visible, n_hidden)

    # Put data in a torch.DataLoader
    # NOTE: Not using this yet
    # v_dataloader = torch.utils.data.DataLoader(v_data, batch_size=batch_size, shuffle=True, num_workers=0)

    rbm.train(v_data_probs, epochs=epochs)


if __name__ == "__main__":
    uniform_dist_test()