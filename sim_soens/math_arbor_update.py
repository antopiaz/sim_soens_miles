import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import loadtxt
import networkx as nx
from math_sim import generate_graph, get_leaves, s_of_phi, plot_signal_flux

t=250
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
        weight_matrix[i][12]+=0.5
print(weight_matrix)



def arbor_step(t,n,phi_spd, flux_offset):
    '''
    Iterates through time and updates flux and signal using the equation (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]
    and signal is updated using the update equation (4) from phenom paper
    '''
    plot_signals = np.zeros((t,n))
    plot_fluxes = np.zeros((t,n))
    count = 0


    leaf_nodes = get_leaves(weight_matrix,n,random_val=False)

    signal_vector = leaf_nodes*phi_spd#leaf_nodes*data[0]

    #print('plot ', np.shape(plot_signals))
    #print('matrix ',np.shape(weight_matrix))
    #print('leaf ', (leaf_nodes))
    #print('sig ', np.shape(signal_vector))

    for i in range(t):
        #print('slice ',plot_signals[:])
        #print('avg ',flux_offset)
        flux_vector = (signal_vector@weight_matrix) + leaf_nodes*phi_spd + flux_offset#leaf_nodes*data[i%10000]

        #if(i==162):
        #    print(data[i])
        #    print('leaf checl ', leaf_nodes*data[i])
        #    print('check ', flux_vector)
        #    print('check ',signal_vector@weight_matrix)
        
        signal_vector = signal_vector*(1- (1e-9/1.2827820602389245e-12)*(.053733049288045114/(2*np.pi*1e3))) + ((1e-9/1.2827820602389245e-12)/(2*np.pi*1e3))*s_of_phi(flux_vector, signal_vector,n)
        if signal_vector[-1]>0.7:
            count += 1
            signal_vector[-1]=0
        #dend.s[t_idx+1] = dend.s[t_idx]*(1 - d_tau*dend.alpha/dend.beta) + (d_tau/dend.beta)*r_fq
    
        plot_signals[i] = signal_vector
        plot_fluxes[i] = flux_vector
    
    return plot_signals, plot_fluxes, weight_matrix, count


def arbor_update_rule(letters,  learning_rate=.01):
    convergence = False
    flux_offset = np.zeros(n)
    i=0
    while convergence != True:
        
        expected_spikes =[1,0,0]# [1,2,4]
        total_error = np.zeros(3)


        for k in range(np.shape(letters)[0]):

            
        #print(letters[1])
            training_spikes = [expected_spikes[i], 0 ,0]

            plot_signals, plot_fluxes, weight_matrix, count = arbor_step(t,n,letters[k], flux_offset)

            error=expected_spikes[k]-count #is error same as delta y? how to calculate
            #print(expected_spikes[k])
            #error = training_spikes[k]-count
            total_error[k] = total_error[k]-error

            #print('total ',total_error)

            for j in range(n):
                flux_offset[j] +=  learning_rate*np.average(plot_signals[:,j])*error  
                #* expected_signal[i]-plot_signals[i] #use a running average or an exact average? using plot signals?
                #spikes from soma threshold

                #print(np.shape(plot_signals[:,j]))
                #print(plot_signals[:,j])

            #print('i ',i)
            #if i==0:
            #    plot_signal_flux(plot_signals, plot_fluxes,weight_matrix, t, n)

        if total_error.any()==0:
            convergence=True
            print('flux ',flux_offset)
            print(count)
            #plot_signal_flux(plot_signals, plot_fluxes,weight_matrix, t, n)
            return flux_offset
    
flux = arbor_update_rule(letters)
            
fluxz=  [-0.23624039, -0.22288985, -0.1056304,  -0.19017632, -0.14086677, -0.19017632,
 -0.09149359, -0.23624039, -0.22288985, -0.25441734, -0.33246438, -0.23923557,
 -0.21021379]



flux2= [ 1.20087569,  0.  ,        1.20087569 , 1.20087569 , 0.    ,      1.20087569,
  0.  ,        1.20087569 , 0.        ,  0.32111584 , 0.32111584, -0.33801419,
 -0.33471852]

flux3 =  [ 0.98107378,  0.98107378,  0.      ,    0.     ,    0.98107378 , 0. ,
  0.     ,     0.98107378 , 0.98107378 , 0.1237079 , -0.34240398 , 0.1237079,
 -0.32074186]

plot_signals, plot_fluxes, weight_matrix, count = arbor_step(t,n,letters[2], flux)
print(count)
plot_signal_flux(plot_signals, plot_fluxes,weight_matrix, t, n)

