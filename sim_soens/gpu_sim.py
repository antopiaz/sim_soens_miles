import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import loadtxt
import networkx as nx
from numba import jit, cuda
from numba.cuda import random
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
#from pycuda import driver, compiler, gpuarray, tools
import time
#import pycuda.autoinit
import cupy as cp
from cupy import random



#hello this is a test comment

print(cp.float32(.466))
#phi_spd
data = loadtxt('phi_signal.csv', delimiter=',')
data = cp.float16(data)
plt.plot(np.arange(0,10001), data)
plt.savefig('phi_sig_plot.png', dpi=400, bbox_inches='tight')

#total_time_2 = loadtxt('time_perf2.csv', delimiter=',')
#plt.plot(np.arange(0,50000,1000), total_time_2)

#plt.plot(np.arange(2,1020,30), total_time_1, "r")
#plt.show()

#phi_th=0.1675
phi_th=cp.float32(0.1675)
t=1000

d_tau = 1e-9/1.2827820602389245e-12
d_tau = np.float32(d_tau)

beta = 2*np.pi*1e3
beta = cp.float32(beta) #different by .0002

alpha = 0.053733049288045114
alpha = np.float32(alpha)

A=1
print(type(beta))
B=np.float32(.466)
ib=np.float32(1.8)

mempool = cp.get_default_memory_pool()
pinned_mempool = cp.get_default_pinned_memory_pool()


#@jit(nopython=True)
@cuda.jit
def s_of_phi(phi,s,n, r_fq):
    """
    Function to get rate array 
    """
    
    for i in range(n):
        r_fq[i] = A*(phi[i]-phi_th)*((B*ib)-s[i])
    #r_fq = A*((phi-phi_th)*(((B*ib)-s))) #s vector, and phi
    
    #indices = np.where(phi<phi_th)[0]
    #for i in indices:
    #    r_fq[i] = 0  #if phi from incoming node is below threshold then it passes on nothing
    for i in range(n):
        if phi[i]<phi_th: 
            r_fq[i] = 0
    #return r_fq




#@jit(nopython=True)
def neuron_step(t,n, data):
    '''
    Iterates through time and updates flux and signal using the equation (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]
    and signal is updated using the update equation (4) from phenom paper
    '''
    #phi_spd=0.5
    #learning_rate=.01

    #start_gpu = cp.cuda.Event()
    #end_gpu= cp.cuda.Event() 

    #tp1=time.perf_counter()
    plot_signals =  cp.zeros((t,n), dtype=cp.float16)
    plot_fluxes = 0#cp.zeros((t,n)).astype(np.float32)
    #tp2=time.perf_counter()
    #print('init', tp2-tp1)

    #tw1 = time.perf_counter()
    #weight_matrix = ((cp.random.rand(n,n, dtype=cp.float32))) / (n * 0.5/0.72)
    weight_matrix = ((cp.full((n,n),0.5, dtype=cp.float16))) / (n * 0.5/0.72)
    #weight_matrix = cp.full((n,n), 0.05, dtype=cp.float16)
    cp.fill_diagonal(weight_matrix,0) #eye 
    weight_matrix[n-1] = 0
    weight_matrix[0,1]=1

    #weight_matrix = cp.clip(weight_matrix,0,1)
    #tw2 = time.perf_counter()
    #print('weight time', tw2-tw1)
    #print('weight',(weight_matrix))

    #tl1 = time.perf_counter()
    #leaf_nodes = ((cp.random.rand(n, dtype=cp.float32))-0.95) *(0.5/0.72)

    test1 = cp.ones(int(n*0.05),dtype=cp.float16)
    test2 = cp.zeros(int(n-(n*0.05)),dtype=cp.float16)
    leaf_nodes=cp.append(test1,test2)
    #print(test3)
    #print(cp.count_nonzero(test3))

    leaf_nodes[n-1] = 0
    leaf_nodes[0]=1

    leaf_nodes = cp.clip(leaf_nodes,0,1)
    #tl2 = time.perf_counter()
    #print('leaf_nodes', tl2-tl1)
    print('leaf',cp.dtype(leaf_nodes))

    signal_vector = leaf_nodes*data[0]
    #print('sig', cp.dtype(signal_vector))
    #ta = time.perf_counter()
    print('used bytes',mempool.used_bytes())
    print('total bytes',mempool.total_bytes())
    print('cpu mem?',pinned_mempool.n_free_blocks())
    for i in range(t):
        print(f"Timestep = {i}", end="\r")
       
        #flux_offset =  learning_rate*np.average(plot_signals[:,i]) #* expected_signal[:][i]-plot_signals[:][i] #use a running average or an exact average? using plot signals?
        flux_vector=(cp.matmul(signal_vector, weight_matrix)+(leaf_nodes * data[i%10000]))
        #flux_vector= flux_vector/(cp.max(flux_vector[-1])+1)

        #dend.s[t_idx+1] = dend.s[t_idx]*(1 - d_tau*dend.alpha/dend.beta) + (d_tau/dend.beta)*r_fq
        #signal_vector = cp.add(cp.multiply(signal_vector,(1- (1e-9/1.2827820602389245e-12)*(.053733049288045114/(2*np.pi*1e3)))),cp.multiply(s_of_phi(flux_vector, signal_vector,n), ((1e-9/1.2827820602389245e-12)/(2*np.pi*1e3))))
        
        r_fq = cp.zeros(n, dtype=cp.float16)
        s_of_phi[32,32](flux_vector, signal_vector,n, r_fq)

        #indices = cp.where(r_fq<phi_th)[0]
        #print(indices)

        signal_vector = signal_vector*(1 - d_tau*alpha/beta) + (d_tau/beta )*r_fq
        #cp.clip(signal_vector,0,0.5)

        #if signal_vector[n-1]>0.7:
        #    signal_vector[n-1]=0
        
        #print('flux',flux_vector[200])
        #print('signal',signal_vector[200])
        #print('max',cp.max(signal_vector))
        plot_signals[i] = signal_vector
        #plot_fluxes[i] = flux_vector

    #tb = time.perf_counter()
    #print('runtime', tb-ta)
    print('>< after')
    print('used bytes',mempool.used_bytes())
    print('total bytes',mempool.total_bytes())
    print('cpu mem?',pinned_mempool.n_free_blocks())

    return plot_signals, plot_fluxes, weight_matrix


def time_measure(data, t, mode="length"):
    '''
    Used to measure the performance of neuron step function for increasing time length and increasing dendrites
    '''
    time_array = np.zeros(0)
    count=0
    
    if(mode=="length"):
        #n = random.randint(2,20)
        n=100

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

            #check to make sure it is running correctly
            if(j==50000):

                truncate = 0
                time_axis = np.arange(truncate,49000)

                fig, axs = plt.subplots(n)

                for i in range(n):
                    axs[i].plot(time_axis, plot_signals[:,i][truncate:49000], label='signal')
                    axs[i].plot(time_axis, plot_fluxes[:,i][truncate:49000], label='fluxes')
                    #axs.legend()
                    #axs[i].set_title('node ' + str(i))
                plt.show()

    
    elif (mode=="size"):
        for k in range(2,6020,1000):

            print(f"Run = {k}", end="\r")
            

            start_gpu= cp.cuda.Event()
            end_gpu = cp.cuda.Event()

            start_gpu.record()
            t1 = time.perf_counter()

            out = cp.zeros((k,k))
            graph[32,64](k,out)
            plot_signals,plot_fluxes, weight_matrix = neuron_step(int(t), k , data,out)

            t2 = time.perf_counter()
            end_gpu.record()
            end_gpu.synchronize()

            run_time = t2-t1
            print('gpu', cp.cuda.get_elapsed_time(start_gpu, end_gpu))
            print(run_time)
            print(k)
            

            time_array=np.append(time_array, run_time)
            #print(time_array)

    return time_array



mode=''
if(mode=='size'):
    print(mode)
    total_time = time_measure( data,t, mode="size")
    #np.savetxt("time_perf1.csv", total_time, delimiter=",")

    #print(np.shape(np.arange(2,1020,11)))
    plt.plot(np.arange(2,6020,1000), total_time)
    plt.savefig('test_plot4.png', dpi=400, bbox_inches='tight')
    print(total_time)
    #plt.plot(np.arange(2,1020,30), total_time_1, "r")
    
    plt.show()
elif(mode=='length'):
    total_time = time_measure( data,t, mode="length")
    #np.savetxt("time_perf2.csv", total_time, delimiter=",")
    plt.plot(np.arange(0,t,1000), total_time)
    plt.savefig('test_plot3.png', dpi=400, bbox_inches='tight')
    print(total_time)
    #plt.plot(np.arange(0,t,1000), total_time_2, "r")

    #plt.show()


#plot
def plot_signal_flux(plot_signals, plot_fluxes,weight_matrix, t, n):
    '''
    Plots the signal flux and dendrite graph
    '''
    truncate = 0
    time_axis = np.arange(truncate,t)

    fig, axs = plt.subplots(n)
    #plt.set_ylim(bottom=0)
    #plt.ylim(top=1)
    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.8)
    
    for i in range(n):
        axs[i].plot(time_axis, plot_signals[:,i][truncate:t], label='signal')
        axs[i].plot(time_axis, plot_fluxes[:,i][truncate:t], label='fluxes')
        axs[i].set_ylim(0,1.2)
        #axs.legend()
        axs[i].set_title('node ' + str(i))
    #plt.show()
    plt.savefig('test_plot1.png', dpi=400, bbox_inches='tight')


    #fig, axs = plt.subplots(n)

    #for i in range(n):
    #    axs[i].plot(time_axis, plot_signals[:,i][truncate:t], label='signal')
    #plt.show()


    edges = []
    for i in range(n):
        for j in range(n):
            if weight_matrix[:,i][j] !=0:
                edges.append((j, i))

    #print(edges)
    G = nx.DiGraph(directed=True)
    G.add_edges_from(
        edges)
    
    edge_labels = dict([((n1, n2), np.around(weight_matrix[n1][n2], decimals=2))
                    for n1, n2 in G.edges])
    pos = nx.planar_layout(G)

    nx.draw_networkx(G,pos)#, node_color=values)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    #values = np.ones(n)
    #values[n-1]+=1
    #print(values)
    #pos = nx.planar_layout(G)
    #plt.show()
    plt.savefig('test_plot2.png', dpi=400, bbox_inches='tight')






#test_array = [1,0,1,0,0.5,1,0.6]
#test_array_gpu = cp.asarray(test_array)

print('used bytes',mempool.used_bytes())
print('total bytes',mempool.total_bytes())
print('cpu mem?',pinned_mempool.n_free_blocks())
print('^ before')
k=40000
#tt1 = time.perf_counter()
#test = cp.random.rand(k,k)
#tt2 = time.perf_counter()
#print('cp rand', tt2-tt1)
#print('orig',test)
#cp.fill_diagonal(test,0)
#print('diag',test)
t1 = time.perf_counter()



plot_signals,plot_fluxes, weight_matrix = neuron_step(int(t), k , data)
t2 = time.perf_counter()
print('time',t2-t1)
print('count',cp.count_nonzero(weight_matrix)/(k**2))
print(weight_matrix)
print('sum',cp.sum(weight_matrix[:,1]))
time_axis = np.arange(t)
#plt.plot(time_axis, cp.asnumpy(plot_signals)[:,0], label='fluxes')
#plt.savefig('test_plot4.png', dpi=400, bbox_inches='tight')
#print(cp.asnumpy(plot_fluxes[:,k-1]))
fig, axs = plt.subplots(6)
axs[0].plot(time_axis, cp.asnumpy(plot_signals)[:,0], label='node 0')
axs[1].plot(time_axis, cp.asnumpy(plot_signals)[:,1], label='node 10')
#axs[2].plot(time_axis, cp.asnumpy(plot_signals)[:,100], label='node 100')
#axs[3].plot(time_axis, cp.asnumpy(plot_signals)[:,1000], label='node 1000')
axs[4].plot(time_axis, cp.asnumpy(plot_signals)[:,2500], label='node 2500')
axs[5].plot(time_axis, cp.asnumpy(plot_signals)[:,k-1], label='soma')

#for i in range(6):
    #axs[i].set_ylim(0,1.2)
    #axs[i].set_xlim(0,2000)

plt.savefig('test_plot_scratch1.png', dpi=400, bbox_inches='tight')





    
