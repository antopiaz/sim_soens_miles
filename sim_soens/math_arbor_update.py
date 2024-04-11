import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import loadtxt
import networkx as nx
from math_sim import generate_graph, get_leaves, s_of_phi, plot_signal_flux

t=100
n=13
letters = [np.array([1,1,0,0,1,0,0,1,1,  0,0,0,0]), np.array([1,0,1,1,0,1,0,1,0  ,0,0,0,0]), np.array([0,1,0,1,0,1,1,0,1  ,0,0,0,0]) ] #adding a bunch of zeros
#maybe extend or shorten one of the arrays later on but could be very expensive
letters = np.array(letters)


weight_matrix=np.zeros((13,13))

for i in range(13):
    if i<3:
        weight_matrix[i][9]+=0.5
    if 2<i<6:
        weight_matrix[i][10]+=0.5
    if 5<i<9:
        weight_matrix[i][11]+=0.5
    if 8<i<12:
        weight_matrix[i][12]+=1
print(weight_matrix)



def neuron_step(t,n,phi_spd):
    '''
    Iterates through time and updates flux and signal using the equation (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]
    and signal is updated using the update equation (4) from phenom paper
    '''
    learning_rate=.01
    plot_signals = np.zeros((t,n))
    plot_fluxes = np.zeros((t,n))
    


    leaf_nodes = get_leaves(weight_matrix,n,random_val=False)

    signal_vector = leaf_nodes*phi_spd#leaf_nodes*data[0]
    print('sig ',signal_vector)

    #print('plot ', np.shape(plot_signals))
    #print('matrix ',np.shape(weight_matrix))
    #print('leaf ', (leaf_nodes))
    #print('sig ', np.shape(signal_vector))

    for i in range(t):
        flux_offset =  learning_rate*np.average(plot_signals[:,i])  #* expected_signal[:][i]-plot_signals[:][i] #use a running average or an exact average? using plot signals?
        print('slice ',plot_signals[:,i])
        print('avg ',flux_offset)
        flux_vector = (signal_vector@weight_matrix) + leaf_nodes*phi_spd + flux_offset#leaf_nodes*data[i%10000]

        #if(i==162):
        #    print(data[i])
        #    print('leaf checl ', leaf_nodes*data[i])
        #    print('check ', flux_vector)
        #    print('check ',signal_vector@weight_matrix)
        
        signal_vector = signal_vector*(1- (1e-9/1.2827820602389245e-12)*(.053733049288045114/(2*np.pi*1e3))) + ((1e-9/1.2827820602389245e-12)/(2*np.pi*1e3))*s_of_phi(flux_vector, signal_vector,n)
        
        #dend.s[t_idx+1] = dend.s[t_idx]*(1 - d_tau*dend.alpha/dend.beta) + (d_tau/dend.beta)*r_fq
    
        plot_signals[i] = signal_vector
        plot_fluxes[i] = flux_vector
    
    return plot_signals, plot_fluxes, weight_matrix


def arbor_update_rule(letters):
    convergence = False
    while convergence != True:
        total_error =0
        for i in range(np.size(letters)):
            print(letters[i])
            plot_signals, plot_fluxes, weight_matrix = neuron_step(t,n,letters[i])
            plot_signal_flux(plot_signals, plot_fluxes,weight_matrix, t)

arbor_update_rule(letters)


