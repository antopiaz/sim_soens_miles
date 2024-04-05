import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import loadtxt
import networkx as nx
from numba import jit
import time


#signal
data = loadtxt('phi_signal.csv', delimiter=',')
#plt.plot(np.arange(0,10001), data)
#plt.show()


phi_th=0.1675
t=50000


@jit(nopython=True)
def s_of_phi(phi,s,n,A=1,B=.466,ib=1.8):
    """
    Function to get rate array 
    """
    phi_th = 0.1675
    r_fq = A*(phi-phi_th)*((B*ib)-s) #s vector, and phi     
    for i in range(n):
        if phi[i]<phi_th: 
            r_fq[i] = 0  #if phi from incoming node is below threshold then it passes on nothing
    return r_fq

@jit(nopython=True)
def generate_graph(n):
    '''
    Generate arbitrary tree graph using adjacency graph
    '''
    weight_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(i):  #graph connectivity problems
            if weight_matrix[j].any() == 0: #why does this work?
                value = random.randint(j+1,n-1)   # random.choice(np.arange(j+1,n,1))
                weight_matrix[j][value]=(round(random.random(),2))*0.7+0.3
                #weight_matrix[j][value]=0.5
    
    return weight_matrix

@jit(nopython=True)
def get_leaves(weight_matrix, n):
    '''
    to find columns that are empty implying a leaf node
    '''

    leaf_nodes = np.zeros(n)            
    for i in range(n-1):
        if(weight_matrix[:,i].any()==0):
            leaf_nodes[i]+=(round(random.random(),2))*0.7+0.3
            #leaf_nodes[i]+=0.5

    return(leaf_nodes)


#iterate through time
@jit(nopython=True)
def neuron_step(t,n, data):
    '''
    Iterates through time and updates flux and signal using the equation (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]
    and signal is updated using the update equation (4) from phenom paper
    '''

    plot_signals = np.zeros((t,n))
    plot_fluxes = np.zeros((t,n))
    #print('plot ', np.shape(plot_signals))
    weight_matrix = generate_graph(n)
    #print('matrix ',np.shape(weight_matrix))
    leaf_nodes = get_leaves(weight_matrix,n)
    #print('leaf ', (leaf_nodes))
    signal_vector = leaf_nodes*data[0]
    #print('sig ', np.shape(signal_vector))

    
    for i in range(t):
        flux_vector = (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]
        #if(i==162):
        #    print(data[i])
        #    print('leaf checl ', leaf_nodes*data[i])
        #    print('check ', flux_vector)
        #    print('check ',signal_vector@weight_matrix)
        

        signal_vector = signal_vector*(1- (1e-9/1.2827820602389245e-12)*(.053733049288045114/(2*np.pi*1e3))) + ((1e-9/1.2827820602389245e-12)/(2*np.pi*1e3))*s_of_phi(flux_vector, signal_vector,n)
        
        #dend.s[t_idx+1] = dend.s[t_idx]*(1 - d_tau*dend.alpha/dend.beta) + (d_tau/dend.beta)*r_fq
    
        plot_signals[i] = signal_vector
        plot_fluxes[i] = flux_vector
    
    return plot_signals, plot_fluxes, weight_matrix #, signal_vector


def time_measure(data, t, mode="length"):

    time_array = np.zeros(0)
    count=0
    
    if(mode=="length"):
        n = random.randint(2,20)

        for j in range(0,t,1000):
            t1 = time.perf_counter()
            plot_signals,plot_fluxes, weight_matrix = neuron_step(j,n, data)
            t2 = time.perf_counter()
            run_time = t2-t1
            #print(run_time)
            if(j%1000==0):
                print(count)
                count +=1
            time_array=np.append(time_array, run_time)
    
    elif (mode=="size"):
        for k in range(2,1020,30):
            t1 = time.perf_counter()
            plot_signals,plot_fluxes, weight_matrix = neuron_step(int(t),k, data)
            t2 = time.perf_counter()
            run_time = t2-t1
            

            time_array=np.append(time_array, run_time)
            #print(time_array)

    return time_array


mode='length'#'length'
if(mode=='size'):
    print(mode)
    total_time = time_measure( data,t, mode="size")
    print(np.shape(np.arange(2,1020,11)))
    plt.plot(np.arange(2,1020,30), total_time)
    plt.show()
elif(mode=='length'):
    total_time = time_measure( data,t, mode="length")

    plt.plot(np.arange(0,t,1000), total_time)
    plt.show()



#plot
def plot_signal_flux(plot_signals, plot_fluxes,weight_matrix, t):


    truncate = 0
    time_axis = np.arange(truncate,t)

    fig, axs = plt.subplots(n)

    for i in range(n):
        axs[i].plot(time_axis, plot_signals[:,i][truncate:t], label='signal')
        axs[i].plot(time_axis, plot_fluxes[:,i][truncate:t], label='fluxes')
        #axs.legend()
        #axs[i].set_title('node ' + str(i))
    plt.show()

    #fig, axs = plt.subplots(n)

    #for i in range(n):
    #    axs[i].plot(time_axis, plot_signals[:,i][truncate:t], label='signal')
    #plt.show()


    edges = []
    for i in range(n):
        for j in range(n):
            if weight_matrix[:,i][j] !=0:
                edges.append((j, i))

    print(edges)
    G = nx.DiGraph(directed=True)
    G.add_edges_from(
        edges)

    values = np.ones(n)
    values[n-1]+=1
    print(values)
    pos = nx.planar_layout(G)

    nx.draw_networkx(G,pos=pos)#, node_color=values)


    plt.show()


n=5
plot_signals,plot_fluxes,weight_matrix = neuron_step(t,n, data)

print('weights \n',weight_matrix)
#print('fluxes ',signal_vector@weight_matrix)
#print('signals ', signal_vector)
#print('leaf ', leaf_nodes)
#print('runtime ', run_time)
#print('plot ', plot_signals[:,0])
plot_signal_flux(plot_signals, plot_fluxes,weight_matrix, t)