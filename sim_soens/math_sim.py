import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import loadtxt

#signal
data = loadtxt('phi_signal.csv', delimiter=',')
plt.plot(np.arange(0,10001), data)
plt.show()


phi_th=0.1675
n = random.randint(1,10)
n=6
t=164
weight_matrix = np.zeros((n,n))
plot_signals = np.zeros((t,n))

def s_of_phi(phi,s,A=1,B=.466,ib=1.8):
    """
    Function to get rate array 
    """
    phi_th = 0.1675
    r_fq = A*(phi-phi_th)*(B*ib-s)
    if phi.any()<0.1675: r_fq = np.zeros(n)
    print('calculated ',r_fq)
    return r_fq#np.clip(r_fq,-2,2)



def generate_graph(weight_matrix,n):
    '''
    Generate arbitrary tree graph using adjacency graph
    '''
    for i in range(n):
        for j in range(i):  #graph connectivity problems
            if weight_matrix[j].any() == 0: #why does this work?
                value = random.choice(np.arange(j+1,n,1))   
                weight_matrix[j][value]=round(random.random(),2)
    
    return weight_matrix

weight_matrix = generate_graph(weight_matrix,n)

def get_leaves(weight_matrix, n):
    '''
    to find columns that are empty implying a leaf node
    '''

    leaf_nodes = np.zeros(n)            
    for i in range(n-1):
        if(weight_matrix[:,i].any()==0):
            leaf_nodes[i]+=1
    print('leaf ',leaf_nodes)

    return(leaf_nodes)

leaf_nodes = get_leaves(weight_matrix,n)

#signal vector
signal_vector = leaf_nodes*data[0]

#iterate through time
for i in range(t):
    flux_vector = signal_vector@weight_matrix + leaf_nodes*data[i]
    print('spd ',leaf_nodes*data[i])

    signal_vector = signal_vector*(1- 1/.466) + (1/.466)*s_of_phi(flux_vector, signal_vector)
    #print(s_of_phi(flux_vector, signal_vector))
    plot_signals[i] = signal_vector


#plot
print('weights \n',weight_matrix)
print('fluxes ',signal_vector@weight_matrix)
print('signals ', signal_vector)
print('plot ', plot_signals[:,0])

time_axis = np.arange(0,t)
plt.plot(time_axis, plot_signals[:,n-1])
plt.plot(time_axis+t, plot_signals[:,n-2])
plt.plot(time_axis+2*t, plot_signals[:,n-3])

plt.show()


