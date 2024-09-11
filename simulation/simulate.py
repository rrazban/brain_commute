"""
Code to simulate the mean field Ising model.

To run this script, you need to specify the lambda value in the arguments
i.e. ./simulate.py 2

"""

import numpy as np
import sys, random, os
import matplotlib.pyplot as plt
import csv
import pandas as pd

sys.path.append('../utlts/')
from analysis_tools import delete_indi, no_edges, delete_empty_index



def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('lambdaa', type=float, metavar='LAMBDA', help="units of coupling/kT. It is not normalized (has extra N**2 factor). For weighted graph, lambda is rescaled by the streamline count as determined from dMRI.")
    parser.add_argument('--num-sweeps', type=int, metavar='NUM', default=5000)	#corresponds to the number of measurements
    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def boltzmann_weight(energy, beta):
    return np.exp(energy * beta)


class IsingSystem:
    """ 
    the Mean-Field Ising System
    """
    def __init__(self, size, sweeps, non_normalized_lambda, adjacency):

        self.system = 2*np.random.randint(2, size=size) - 1 #initial spin state randomly set

        self.lambdaa = non_normalized_lambda/size**2

        self.adjacency = adjacency
        self.times = sweeps+1
        self.size = size

        self.magnetization_timeseries = np.zeros(sweeps+1)
        self.energy_timeseries = np.zeros(sweeps+1)
        self.success_timeseries = np.zeros(sweeps+1, dtype=int)        # track number of accepted moves per sweep:
        self.output = np.zeros((sweeps+1, size))


        """
        The following two variables set the number of steps per sweep and the number of attempted spin flips per step 
        They are important to get the simulation to quickly reach equilibrium
            -we need to reach equilibrium if we want to compare to mathematical equations because they are derived at equilibrium
        Their exact values do not matter, as long as equilibrium is reached
        """
        self.steps_per_sweep = size #number of steps before record spin state
        self.num_attempts = int(0.15*size)#10	#num of spins attempted to be flipped at a given time step
		
        return

    def calculate_magnetization(self):	#otherwise known as synchrony when taking the mean
        return np.sum(self.system)

    def calculate_energy(self, spins):
        """
        The full energy function involves calculation of all pairwise
        energy interactions. Assume spins are in contact according to the adjacency matrix 
        """

        energy = self.lambdaa * np.dot(spins, np.dot(self.adjacency, spins))
        return energy       #even though double counting of edges, do not divide by 2 to be consistent with fully connected ising model formalizm
    
    def calculate_deltaE(self, positions):
        """
        Position should be an index corresponding to a region.
        """
        new_spins = np.copy(self.system)
        new_spins[positions] *= -1

        deltaE = self.calculate_energy(new_spins) - self.calculate_energy(self.system)
        return deltaE
            

    def run_simulation(self, verbose=False):
        """
        Implements a Metropolis-Hastings algorithm to determine whether 
        attempted sets of spin flips are accepted or not
        """


        """initialize output variables"""
        self.output[0] = self.system
        self.magnetization_timeseries[0] = self.calculate_magnetization()
        self.energy_timeseries[0] = self.calculate_energy(self.system)
        self.success_timeseries[0] = 0
        if verbose:
            print("{0:>10} {1:>15} {2:>15} {3:>15}".format(
                "Time", "Magnetization", "Energy", "NumSuccesses"))
            print("{0:10d} {1:15.3f} {2:15.3f} {3:15d}".format(
                0, self.magnetization_timeseries[0],
                self.energy_timeseries[0], self.success_timeseries[0]))

        """let the dynamics begin"""
        for time in range(1,self.times):
            num_successes = 0
            for j in range(self.steps_per_sweep):
                selected_spins = np.random.randint(self.size, size=self.num_attempts)

                deltaE = self.calculate_deltaE(selected_spins)
                """Metropolis-Hastings algorithm"""
                if deltaE > 0:
                    self.system[selected_spins] *= -1
                    num_successes += 1
                else:
                    if np.random.rand() < boltzmann_weight(deltaE, 1):
                        self.system[selected_spins] *= -1
                        num_successes += 1

            """record output values"""
            self.magnetization_timeseries[time] = self.calculate_magnetization()
            self.energy_timeseries[time] = self.calculate_energy(self.system)
            self.success_timeseries[time] = num_successes
            self.output[time] = self.system
            if verbose:
                if num_successes>0: 
                	print("{0:10d} {1:15.3f} {2:15.3f} {3:15d}".format(
                    	time, self.magnetization_timeseries[time],
                    	self.energy_timeseries[time], self.success_timeseries[time]))


        return

def writeout_spin_states(out_dir, lam, sim_id, output):
	filename = 'lam{0}_sim{1}.csv'.format(lam, sim_id)
	print('output filename: {0}'.format(filename))
	with open("{0}/{1}".format(out_dir, filename), "w", newline='') as f:
		writer = csv.writer(f)
		writer.writerows(output) 

def hrf(t):
    "A hemodynamic response function"
    return t ** 8.6 * np.exp(-t / 0.547)

def writeout_bold(out_file, outputs, times, size):
        outputs = np.array(outputs)

        dt = 0.1
        hrf_times = np.arange(0, 20, 0.1)
        hrf_signal = hrf(hrf_times)
        extra_times = np.arange(len(hrf_times)-1) * dt + times*dt
        times = np.arange(0, times, 1)*dt

        outputs[outputs==-1] = 0    #silent, not affected by hrf

        times_and_tail = np.concatenate((times,extra_times))

        bolds = np.zeros((len(times_and_tail), size))
        for i in range(size):
            bolds[:,i] = np.convolve(outputs[:,i], hrf_signal)


        with open(out_file, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerows(bolds[len(hrf_times):-len(hrf_times)])   #remove times where still equilibrating with hrf
 



def main(args,ver):
    sub_id = '1737411'	#'1000366'
    atlas = 'DesKi'

    dmri_file = '../data/{0}/dMRI/{1}_20250_2_0_density.txt'.format(atlas, sub_id)       
    structure = np.loadtxt(dmri_file)
    np.fill_diagonal(structure,0)   #very important, otherwise self-loops are included in calculation of adjacency matrix

    structure, _ = delete_empty_index(structure, structure, atlas)	#second input in fxn should be fc, but not relevant here

    indi = no_edges(structure)
    adjacency = delete_indi(indi, structure)

    size = adjacency.shape[0]
	

    print("# Initializing Mean-Field ising system with %d nodes" % (
        size))
    print("# Will run at lam = %.3f for %d sweeps" % (args.lambdaa, args.num_sweeps))


    ising_system = IsingSystem(size, args.num_sweeps,
                               args.lambdaa, np.copy(adjacency))
    ising_system.run_simulation(args.verbose)

    sim_id = 0
    out_dir = 'output/{0}/{1}'.format(atlas, sub_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = '{0}/bold_lam{1}_sim{2}.csv'.format(out_dir, args.lambdaa, sim_id)
    if os.path.exists(out_file):
        for i in range(200):
            sim_id+=1
            out_file = '{0}/bold_lam{1}_sim{2}.csv'.format(out_dir, args.lambdaa, sim_id)
            if not os.path.exists(out_file):
                break


#    writeout_spin_states(out_dir, args.lambdaa, sim_id, ising_system.output)	#raw Ising signal
    writeout_bold(out_file, ising_system.output, ising_system.times, size)	#convolved Ising signal with hemodynamic response function to imitate BOLD
    if True:   #set to true to check out simulated synchrony distribution to make sure in equilibrium regime (symmetric distribution)
            plt.rcParams.update({'font.size': 14})
            plt.hist(ising_system.magnetization_timeseries/size, range=(-1,1))
            plt.title('subject {0}, $\\lambda = {1}$'.format(sub_id, args.lambdaa))
            plt.xlabel('synchrony')
            plt.ylabel('frequency')
            plt.tight_layout()
            plt.savefig('{0}/synch_lam{1}_sim{2}.png'.format(out_dir,args.lambdaa, sim_id))	#save this in corresponding directory
            plt.show()
    

if __name__ == "__main__":
    main(parse_args(),'default_name')
