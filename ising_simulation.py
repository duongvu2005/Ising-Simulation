# IMPORTS
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import conv2d, pad
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# CHOOSING DEVICE
DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
DEVICE = torch.device('cpu')


# SIMULATION CLASS
class Simulation:
    """
    Creates a 2D Ising Simulation
    """
    def __init__(self, arr_size: int, temp: float, num_sims: int, num_steps: int):
        """
        Parameters:
            - arr_size (int): the size of our Ising array.
            - temp (float): the temperature of the simulation
            - num_sims (int): the number of simulations at each temperature
            - num_steps (int): the number of step of each simulation
        """
        self.N = arr_size
        self.T = temp
        self.sims = num_sims
        self.steps = num_steps

    def create_lattices(self):
        """
        This will create an array with 1 and -1 of the following shape
            [self.sims, 1, self.N, self.N]
        where
        - The self.sims is equal to the number of simulations.
        - The 1 corresponds to 1 channel (the value of the spin)
        - The self.N x self.N corresponds to the size of the array.
        This is the initial lattices of our simulation.
        """
        return torch.stack([
            torch.randint(low=0, high=2, size=(self.N, self.N), dtype=torch.float32) * 2 - 1
            for _ in range(self.sims)
        ]).unsqueeze(dim=1).to(DEVICE)

    def energy_arr(self, lattices):
        kernel = torch.tensor(
            [[0, 1, 0],
             [1, 0, 1],
             [0, 1, 0]], dtype=torch.float32
        ).unsqueeze(dim=0).unsqueeze(dim=0).to(DEVICE)
        energy_arr = -lattices * conv2d(pad(lattices, pad=(1, 1, 1, 1), mode='circular'), kernel, )
        return energy_arr

    def metropolis(self, lattices):
        """
        Do the metropolis algorithm.
        Return the energies and the magnetization of each step (shape: [self.steps, self.sims])
        """
        energies = []
        magnetizations = []
        betas = (
            torch.tensor([1 / self.T for _ in range(self.sims)])
        ).reshape([-1, 1, 1, 1]).to(DEVICE)

        for _ in tqdm(range(self.steps), desc="Loading..."):
            i = np.random.randint(0, 2)
            j = np.random.randint(0, 2)
            energy_arr = self.energy_arr(lattices)
            energies.append(energy_arr.sum(axis=(1, 2, 3)) / 2 / self.N**2)
            magnetizations.append(torch.abs(lattices.sum(axis=(1, 2, 3))) / self.N**2)
            delta_E = -2 * energy_arr
            probs = torch.exp(-betas * delta_E)
            flip_arr = (delta_E > 0) * (torch.rand(delta_E.shape).to(DEVICE) < probs) + (delta_E <= 0)
            lattices[..., i::2, j::2][flip_arr[..., i::2, j::2]] *= -1
        return torch.vstack(energies), torch.vstack(magnetizations)


if __name__ == '__main__':
    pass
