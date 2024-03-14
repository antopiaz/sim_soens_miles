import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../sim_soens')
sys.path.append('../')
from sim_soens.neuron_library import NeuralZoo
from sim_soens.super_input import SuperInput
from sim_soens.soen_components import input_signal, synapse, neuron, network
from sim_soens.soen_plotting import raster_plot
from sim_soens.super_node import SuperNode

import numba
from numba import jit
from sim_soens.soen_numba_stepper import *

display=True


def simple_net():
    runs =1
    tf = 1000
    spike_times = np.arange(0,tf,100) 
    input_ = SuperInput(type='defined', defined_spikes=spike_times)

    dt = .1
    beta    = 2*np.pi*10**3
    s_th = 0.5
    dct1 = {
        "weights":[[[.5,.5]]],
        "beta"      :beta,
        "beta_ni"   :beta,
        "beta_di"   :beta,
        "s_th"      :s_th,
    }
    

    


    node1 = SuperNode(name='node1', **dct1)


    node2 = SuperNode(name='node2',**dct1)

    for syn in node2.synapse_list:
        print(syn.name)
        node1.neuron.add_output(syn)
    # print("SYNAPSE NAME",node2.synapse_list[0].name)

    node1.uniform_input(input_)

    

    net = network(
        sim         = True,            
        tf          = tf,  
        nodes       = [node1],          
        backend     = "simple",
        print_times = True,
        dt          = 0.1
        )
   
    if(display==True):
        #print(node2.synapse_list[0].spike_times_converted)

        #raster_plot(net.spikes)

        #node1.plot_structure()
        #node1.parameter_print()

        #node2.parameter_print()
        node1.plot_neuron_activity(net=net,phir=True,spikes=False,ref=True,dend=True)

        node1.plot_arbor_activity(net=net,phir=True)

        #node2.plot_neuron_activity(net=net,phir=True,spikes=False,ref=True,dend=True)

        #print("components \n")
        #print(node2.synapse_list[0].__dict__)
        #plt.plot(node1.synapse_list[0].phi_spd)
        #plt.show()
       
        plt.show()


simple_net()
#simple_net()
