import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import loadtxt
import networkx as nx
from numba import jit, cuda
import time
#import cupyx.scipy.sparse
import scipy
import cupy as cp
from cupy import random
from cupy import cuda as cua

#np.set_printoptions(threshold=np.inf)

#phi_spd
data = loadtxt('phi_signal.csv', delimiter=',')
data = cp.float16(data)
#plt.plot(np.arange(0,10001), data)
#plt.savefig('input_sig.png', dpi=400, bbox_inches='tight')

#physical constants
phi_th=0.1675
d_tau = cp.float32(1e-9/1.2827820602389245e-12)
beta = cp.float32(2000*np.pi)
alpha = cp.float32(0.053733049288045114)
A=cp.float32(1)
B=cp.float32(.466)
ib=cp.float32(1.8)

#mempool = cp.get_default_memory_pool()
#pinned_mempool = cp.get_default_pinned_memory_pool()

#time and size parameters
t=2000
k=700 #same as n
neuron_size = 7

@cuda.jit
def s_of_phi(phi,s,n, r_fq):
    """
    Function to approximate rate array 
    """
    for i in range(n):
        if phi[i]<phi_th: 
            r_fq[i] = 0
        else:
            r_fq[i] = A*(phi[i]-phi_th)*((B*ib)-s[i])
        
@cuda.jit
def spike_check(signal_vector, somas, spike_check_arr):
    """
    Iterates through all the soma's to check if their signal is above threshold
    """
    for j in range(int(k/neuron_size)):
        if signal_vector[somas[j]]>0.7:
            signal_vector[somas[j]]=0
            spike_check_arr[somas[j]]=1
    


test = cp.full((k,k), 0, dtype=cp.float16)
mini = [[0, 0.6, 0, 0,   0,   0,   0],
        [0, 0,   0, 0.5, 0,   0,   0],
        [0, 0,   0, 0.7, 0,   0,   0],
        [0, 0,   0, 0,   0.7, 0,   0],
        [0, 0,   0, 0,   0,   0.5, 0],
        [0, 0,   0, 0,   0,   0,   0.7],
        [0, 0,   0, 0,   0,   0,   0,]]
mini = cp.asarray(mini)

for i in range(0,k,neuron_size):
    test[i:i+neuron_size, i:i+neuron_size]=mini


som = cp.zeros(int(k/neuron_size), dtype=int)
som[0] = neuron_size-1
for i in range(1,int(k/neuron_size)):
    som[i]=(som[i-1]+neuron_size)

def neuron_step(t,n, data):
    '''
    Iterates through time and updates flux and signal using the equation (signal_vector@weight_matrix) + leaf_nodes*data[i%10000]
    and signal is updated using the update equation (4) from phenom paper
    '''
    spike_check_arr = cp.zeros(n, dtype=int)
    t_spike = cp.zeros(n, dtype=int)
    somas = som

    start_gpu = cp.cuda.Event()
    end_gpu= cp.cuda.Event() 
    
    plot_signals =  cp.zeros((t,n), dtype=cp.float16)
    plot_fluxes = cp.zeros((t,n), dtype=cp.float16)

    #weight_matrix = ((cp.random.rand(n,n, dtype=cp.float32))) / (n * 0.5/0.72)
    weight_matrix = test
    cp.fill_diagonal(weight_matrix,0) #eye 
    weight_matrix[n-1] = 0

    #leaf_nodes = ((cp.random.rand(n, dtype=cp.float32))-0.95) *(0.5/0.72)
    leaf_nodes = cp.zeros(k)
    leaf_nodes[0:k:neuron_size]= 0.7
    leaf_nodes[2:k:neuron_size]= 0.7
    leaf_nodes[n-1] = 0

    signal_vector = leaf_nodes*data[0]
    #print('used bytes',mempool.used_bytes())
    #print('total bytes',mempool.total_bytes())
    #print('cpu mem?',pinned_mempool.n_free_blocks())
    print(weight_matrix)
    #print(leaf_nodes)

    for i in range(t):
        #print(f"Timestep = {i}", end="\r") 

        #flux_offset =  learning_rate*np.average(plot_signals[:,i]) #* expected_signal[:][i]-plot_signals[:][i] #use a running average or an exact average? using plot signals?
        flux_vector=(cp.matmul(signal_vector, weight_matrix))+(leaf_nodes * data[i%10000])

        if(cp.max(spike_check_arr)==1):
            for x in cp.where(spike_check_arr==1)[0]:
                if  t_spike[x]<20:
                    flux_vector[(x+2)%(k-1)]+=0.5
                    t_spike[x] +=1
                else:
                    spike_check_arr[x] = 0
                    t_spike[x]=0

        #dend.s[t_idx+1] = dend.s[t_idx]*(1 - d_tau*dend.alpha/dend.beta) + (d_tau/dend.beta)*r_fq
        #signal_vector = cp.add(cp.multiply(signal_vector,(1- (1e-9/1.2827820602389245e-12)*(.053733049288045114/(2*np.pi*1e3)))),cp.multiply(s_of_phi(flux_vector, signal_vector,n), ((1e-9/1.2827820602389245e-12)/(2*np.pi*1e3))))

        r_fq = cp.zeros(n, dtype=cp.float16)
        s_of_phi[32,32](flux_vector, signal_vector,n, r_fq)
        signal_vector = signal_vector*(1 - d_tau*alpha/beta) + (d_tau/beta )*r_fq
            
        start_gpu.record()

        if (cp.max(signal_vector[somas])>0.7):
            spike_check[32,32](signal_vector, somas, spike_check_arr)
       
        end_gpu.record()
        end_gpu.synchronize()
        t_gpu1 = cua.get_elapsed_time(start_gpu, end_gpu)
        print(t_gpu1)            
       
        plot_signals[i] = signal_vector
        plot_fluxes[i] = flux_vector 
        
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



#print('used bytes',mempool.used_bytes())
#print('total bytes',mempool.total_bytes())
#print('cpu mem?',pinned_mempool.n_free_blocks())

start_gpu1 = cp.cuda.Event()
end_gpu1= cp.cuda.Event() 

start_gpu1.record()

plot_signals,plot_fluxes, weight_matrix = neuron_step(t, k , data)

end_gpu1.record()
end_gpu1.synchronize()
t_gpu = cua.get_elapsed_time(start_gpu1, end_gpu1)


print('time',t_gpu/1000)
print('count',cp.count_nonzero(weight_matrix)/(k**2))
#print(weight_matrix[:,1500])
print('sum',cp.sum(weight_matrix[:,1]))
time_axis = np.arange(t)
#plt.plot(time_axis, cp.asnumpy(plot_signals)[:,0], label='fluxes')
#plt.savefig('test_plot4.png', dpi=400, bbox_inches='tight')
#print(cp.asnumpy(plot_fluxes[:,k-1]))
fig, axs = plt.subplots(12)

axs[0].plot(time_axis, cp.asnumpy(plot_fluxes)[:,0])
axs[1].plot(time_axis, cp.asnumpy(plot_fluxes)[:,1])
axs[2].plot(time_axis, cp.asnumpy(plot_fluxes)[:,2])
axs[3].plot(time_axis, cp.asnumpy(plot_fluxes)[:,3])
axs[4].plot(time_axis, cp.asnumpy(plot_fluxes)[:,4])
axs[5].plot(time_axis, cp.asnumpy(plot_fluxes)[:,6])

axs[6].plot(time_axis, cp.asnumpy(plot_fluxes)[:,k-6])
axs[7].plot(time_axis, cp.asnumpy(plot_fluxes)[:,k-5])
axs[8].plot(time_axis, cp.asnumpy(plot_fluxes)[:,k-4])
axs[9].plot(time_axis, cp.asnumpy(plot_fluxes)[:,k-3])
axs[10].plot(time_axis, cp.asnumpy(plot_fluxes)[:,k-2])
axs[11].plot(time_axis, cp.asnumpy(plot_fluxes)[:,k-1])


axs[0].plot(time_axis, cp.asnumpy(plot_signals)[:,0])
axs[1].plot(time_axis, cp.asnumpy(plot_signals)[:,1])
axs[2].plot(time_axis, cp.asnumpy(plot_signals)[:,2])
axs[3].plot(time_axis, cp.asnumpy(plot_signals)[:,3])
axs[4].plot(time_axis, cp.asnumpy(plot_signals)[:,4])
axs[5].plot(time_axis, cp.asnumpy(plot_signals)[:,6])

axs[6].plot(time_axis, cp.asnumpy(plot_signals)[:,k-6])
axs[7].plot(time_axis, cp.asnumpy(plot_signals)[:,k-5])
axs[8].plot(time_axis, cp.asnumpy(plot_signals)[:,k-4])
axs[9].plot(time_axis, cp.asnumpy(plot_signals)[:,k-3])
axs[10].plot(time_axis, cp.asnumpy(plot_signals)[:,k-2])
axs[11].plot(time_axis, cp.asnumpy(plot_signals)[:,k-1])

axs[0].set_title('node ' + str(1))
for i in range(12):
    axs[i].set_ylim(0,1.2)
    axs[i].set_xlim(0,2000)

plt.savefig('flux_sig_spike_plot.png', dpi=400, bbox_inches='tight')