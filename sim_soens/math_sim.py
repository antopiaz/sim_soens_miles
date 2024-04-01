import numpy as np
import random
import matplotlib.pyplot as plt
from numpy import loadtxt
import networkx as nx


#signal
data = loadtxt('phi_signal.csv', delimiter=',')
#plt.plot(np.arange(0,10001), data)
#plt.show()


phi_th=0.1675
n = random.randint(1,20)
n=18
t=1000
weight_matrix = np.zeros((n,n))
plot_signals = np.zeros((t,n))

def s_of_phi(phi,s,A=1,B=.466,ib=1.8):
    """
    Function to get rate array 
    """
    phi_th = 0.1675
    r_fq = A*(phi-phi_th)*(B*ib-s)
    if phi.any()<0.1675: r_fq = np.zeros(n)
    #print('calculated ',r_fq)
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
            leaf_nodes[i]+=round(random.random(),2)
    print('leaf ',leaf_nodes)

    return(leaf_nodes)

leaf_nodes = get_leaves(weight_matrix,n)

#signal vector
signal_vector = leaf_nodes*data[0]

#iterate through time
for i in range(t):
    flux_vector = signal_vector@weight_matrix + leaf_nodes*data[i]
    #print('spd ',leaf_nodes*data[i])

    signal_vector = signal_vector*(1- 1/.466) + (1/.466)*s_of_phi(flux_vector, signal_vector)#np.clip(signal_vector*(1- 1/.466) + (1/.466)*s_of_phi(flux_vector, signal_vector), -1,1)
    #print(s_of_phi(flux_vector, signal_vector))
    plot_signals[i] = signal_vector


#plot
print('weights \n',weight_matrix)
print('fluxes ',signal_vector@weight_matrix)
print('signals ', signal_vector)
#print('plot ', plot_signals[:,0])

truncate = 150
time_axis = np.arange(truncate,t)


fig, axs = plt.subplots(n)

for i in range(n):
    axs[i].plot(time_axis+(i+1)*(t-150), plot_signals[:,i][truncate:t])
    #axs[i].set_title('node ' + str(i))


fig.tight_layout()
plt.show()

edges = []
for i in range(n):
    for j in range(n):
        if weight_matrix[:,i][j] !=0:
            edges.append((j, i))

print(edges)
G = nx.DiGraph(directed=True)
G.add_edges_from(
    edges)

#val_map = {'A': 1.0,
##           'D': 0.5714285714285714,
#           'H': 0.0}

#values = [val_map.get(node, 0.25) for node in G.nodes()]
values = np.ones(n)
values[n-1]+=1
print(values)
pos = nx.planar_layout(G)


nx.draw_networkx(G,pos=pos)#, node_color=values)

plt.show()



