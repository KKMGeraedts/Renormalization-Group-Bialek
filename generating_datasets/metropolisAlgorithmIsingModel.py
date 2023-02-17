import numpy as np
import matplotlib.pyplot as plt
import time

def metropolis_algorithm_2dising(N, T, lattice, E_init, M, neighbours, sweeps=1):
    # Initialize
    obs = np.zeros((N, 4))
    E = np.zeros(N+1)
    E[0] = E_init
    x_rand = []
    x_rand2 = []
    samples = []

    # Run metropolis algorithm
    positions = np.random.randint(len(lattice), size=(N, sweeps, 2))
    for i in range(N):
        for j in range(sweeps):
            # Pick a random spin to flip
            pos = positions[i, j]

            # Update the energy
            deltaE = update_energy(lattice, pos, neighbours)

            # Accept/Reject move based on Boltzmann weight
            if deltaE <= 0:
                lattice[pos[0], pos[1]] *= -1
                M += 2 * lattice[pos[0], pos[1]]
            elif np.random.rand() <= np.exp(-deltaE/T):
                lattice[pos[0], pos[1]] *= -1
                M += 2 * lattice[pos[0], pos[1]]
            else:
                deltaE = 0
            E_init += deltaE

        # Update energy and magnetization
        E[i+1] = E_init
        
        # Save state
        samples.append(np.array(lattice).reshape(-1))
        
        # Compute observables
        obs[i] = compute_observables(lattice, E[i+1], M)
    return lattice, E, obs, M, samples

def compute_observables(lattice, E, M):
    """
    Compute the 4 observables required by the exercise.
    """
    obs = []
    N = len(lattice)**2
    # obs.append(E / N)
    # obs.append(np.sum(lattice) / N)

    # print(np.sum(lattice) / N, M / N)
    # obs.append(abs(np.sum(lattice)) / N)
    # obs.append(np.sum(lattice)**2)
    obs.append(E / N) # Average energy per site
    obs.append(M / N) # Average magnetization per site
    obs.append(abs(M) / N) # Order parameter
    obs.append(M**2) # Magnetization square
    return obs

def update_energy(lattice, pos, neighbours):
    """
    Compute the energy cost of flipping the spin at positions pos.
    """
    # Initialize
    L = len(lattice)
    s = -1 * lattice[pos[0], pos[1]]
    deltaE = 0

    # Change energy
    for n in neighbours:
        deltaE += s * lattice[(pos[0] + n[0]) % L, (pos[1] + n[1]) % L]
    return -2*deltaE

def compute_energy(lattice, J):
    """
    Compute the energy lattice of a square lattice. H = -J sum spin_i*spin_j with the sum 
    going over the neighbours indicated by the relative indices in 'neighbours'.

    Return:
        Energy of the system
    """
    # Initialize
    L = len(lattice)

    # Compute energy
    E = -J * np.sum(lattice * (np.roll(lattice, 1, axis=1)
                + np.roll(lattice, -1, axis=1)
                + np.roll(lattice, 1, axis=0) 
                + np.roll(lattice, -1, axis=0)))
    return E

if __name__ == "__main__":
    # Initial values
    L = 16
    T = 3
    J = 1
    N_thermal = 1_000
    N_mc = 2**14
    neighbours = [[0, 1], [0, -1], [1, 0], [-1, 0]]

    # Set seed
    seed = 100
    np.random.seed(seed)

    # Initialize lattice and compute its energy and magnetization
    lattice = np.random.choice([-1, 1], size=(L, L))
    E_init = compute_energy(lattice, J)
    M_init = np.sum(lattice)

    # Time code
    t_start = time.time()

    # Thermalize system
    lattice, E, _, M, _ = metropolis_algorithm_2dising(N_thermal, T, lattice, E_init, M_init, neighbours, sweeps=L**2)
    
    # Print Thermalizing time
    print(f"Thermalization toke {time.time() - t_start} seconds.")

    # Run metropolis algorithm
    lattice, E, obs, M, samples = metropolis_algorithm_2dising(N_mc, T, lattice, E[-1], M, neighbours, sweeps=L**2)
    
    # Save the samples to npy file
    np.save("ising_model_samples.npy", samples)
    
    # Print running time
    print(f"Running time = {time.time() - t_start}") 

    # Compute average of observables
    print(f"Metropolis algorithm results after N = {N_mc} steps with T={T} and L={L}")
    print("===================================================")
    print(f"Energy per site = {np.mean(obs[:, 0])} +/- {round(np.std(obs[:, 0])/np.sqrt(len(obs[:, 0])-1.), 6)}")
    print(f"Magnetization per site = {np.mean(obs[:, 1])} +/- {round(np.std(obs[:, 1])/np.sqrt(len(obs[:, 1])-1.), 6)}")
    print(f"Order parameter = {np.mean(obs[:, 2])} +/- {round(np.std(obs[:, 2])/np.sqrt(len(obs[:, 2])-1.), 6)}")
    print(f"Magnetization squared = {np.mean(obs[:, 3])} +/- {round(np.std(obs[:, 3])/np.sqrt(len(obs[:, 3])-1.), 6)}")
    print("")
    print("Reference values for T=3 and L=4")
    print("===================================================")
    print(f"Energy per site = -1.017786 +/- 0.000374")
    print(f"Magnetization per site = 0")
    print(f"Order parameter = 0.601592 +/- 0.0002")
    print(f"Magnetization squared = 115.554048 +/- 0.058112")

    # plt.matshow(lattice)
    # plt.title(f"Lattice after N = {N_mc} metropolis steps")
    # plt.show()
