import numpy as np
import numpy as np
#import time
#import numba
#from numba import jit, njit

def simple_net_step(net):
    #phi = np.zeros(4)
    #print(net.time_params)
    for t in range(len(net.time_params['tau_vec'])-1):
        present_time=net.time_params['tau_vec'][t-1]
        count = 0

        for node in (net.nodes):
            neuron = node.neuron
            sum = 0
            for dend in node.dendrite_list:
            
                for i in range(1):
                    #print((dend.name))
                    dend.phi = dend_update(dend, present_time)
                    #print(dend.s)
                    dend.s = (1/.466)*s_of_phi(dend.phi, dend.s) + dend.s*(1-(1/.466))
                    count = count+1
                
                if not hasattr(dend, 'is_soma'):
                    #print(dend.name)
                    sum += dend.s*0.5
                    print(sum)

                if hasattr(dend, 'is_soma'):
                    dend.s = sum
                    
    
           
    print(count)
    print()
    return net


def syn_update(syn):
    
    return 0

def dend_update(dend, present_time):
    '''
    why is the first synapse just empty?, 
    that's why we need to do the whole silly syn_key in syn_inputs
    '''
    phi=0
    for syn_key in dend.synaptic_inputs:
        syn = dend.synaptic_inputs[syn_key]
        #print(syn.name)
        #print(present_time)
        #print((syn.spike_times_converted[:]))

        _st_ind = np.where( present_time > syn.spike_times_converted[:] )[0]

        if(len(_st_ind)>0):
            _st_ind = int(_st_ind[-1])

            phi = spd_response(0.5,0.02,50,0.04,_st_ind)

        



        #print(_st_ind)



    return phi



def find_phi_th(val,A,B):
    return A*np.arccos(val/2) + B*(2-val)

def s_of_phi(phi,s,A=1,B=.466,ib=1.8):
    # phi_th = find_phi_th(ib-s,.540,B)
    phi_th = -0.25*(ib-s) + 0.7
    r_fq = A*(phi-phi_th) #*(ib-s))
    # if phi < phi_th: 
    #     r_fq = 0

    if type(r_fq) == type(np.arange(0,10,1)):
        r_fq[phi<phi_th] = 0.0
    else:
        if phi<phi_th: r_fq = 0
    return r_fq


def spd_response(phi_peak,tau_rise,tau_fall,hotspot_duration,t):

    if t <= hotspot_duration:
        phi = phi_peak * ( 
            1 - tau_rise/tau_fall 
            ) * ( 1 - np.exp( -t / tau_rise ) )
    elif t > hotspot_duration:
        phi = phi_peak * ( 
            1 - tau_rise/tau_fall 
            ) * (
            1 - np.exp( -hotspot_duration / tau_rise )
            ) * np.exp( -( t - hotspot_duration ) / tau_fall )
    
    return phi


'''
  for node in net.nodes:
            #node = net.nodes[j]
            #neuron = net.nodes[j].neuron
            neuron = net.nodes

            # update all input synapses and dendrites       
            #for k in range(len(node.dendrite_list)):
            #    dend = net.nodes[j].dendrite_list[k]
            for dend in node.dendrite_list:
                # if hasattr(dend,'is_soma') and dend.threshold_flag == True:
                #if i==1:
                    #print((dend.__dict__.keys()))
                    #print(dir(dend))
                #if hasattr(dend, "is_soma"):        
                #    print("ref ",dend.absolute_refractory_period_converted)
                #print("phi_r ", dend.phi_r)
                #print("sf  ", dend.self_feedback_coupling_strength)
                numba_dendrite_updater(dend,i,tau_vec[i+1],d_tau)

'''



