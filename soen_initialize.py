import numpy as np
import pickle

from soen_utilities import (
    dend_load_rate_array, 
    dend_load_arrays_thresholds_saturations, 
    index_finder
)


d_params_ri = dend_load_arrays_thresholds_saturations('default_ri')
d_params_rtti = dend_load_arrays_thresholds_saturations('default_rtti')
ib__vec__ri = np.asarray(d_params_ri['ib__list'][:])
ib__vec__rtti = np.asarray(d_params_rtti['ib__list'][:])


def dendrite_drive_construct(dend_obj,tau_vec,t_tau_conversion,d_tau):
    # print("  Constructing:",dend_obj.name)          
    dend_obj.phi_r_external__vec = np.zeros([len(tau_vec)]) # from external drives
    dend_obj.phi_r = np.zeros([len(tau_vec)]) # from synapses and dendrites
    dend_obj.s = np.zeros([len(tau_vec)]) # output variable
    dend_obj.beta = dend_obj.circuit_betas[-1]
    
    # add external drives to this dendrite if they're present
    dend_obj = construct_dendritic_drives(dend_obj) 
    
    # turn external drives to this dendrite into flux
    dend_obj.phi_r_external__vec[:] = dend_obj.offset_flux
    for external_input in dend_obj.external_inputs:
        dend_obj.phi_r_external__vec += dend_obj.external_inputs[external_input].drive_signal * dend_obj.external_connection_strengths[external_input]
        
    # prepare somas for absolute refractory period
    if hasattr(dend_obj, 'is_soma'):
        dend_obj.absolute_refractory_period_converted = dend_obj.absolute_refractory_period * t_tau_conversion
        
    # normalize inputs
    if dend_obj.normalize_input_connection_strengths:        
        J_ij_e__init = 0 # excitatory
        J_ij_i__init = 0 # inhibitory
        J_ij_e = dend_obj.total_excitatory_input_connection_strength
        J_ij_i = dend_obj.total_inhibitory_input_connection_strength
        # print('J_ij = {}'.format(J_ij))
        for external_input in dend_obj.external_inputs:
            if dend_obj.external_connection_strengths[external_input] >= 0:
                J_ij_e__init += dend_obj.external_connection_strengths[external_input]
            elif dend_obj.external_connection_strengths[external_input] < 0:
                J_ij_i__init += dend_obj.external_connection_strengths[external_input]
        for synapse in dend_obj.synaptic_inputs:
            if dend_obj.synaptic_connection_strengths[synapse] >= 0:
                J_ij_e__init += dend_obj.synaptic_connection_strengths[synapse]
            elif dend_obj.synaptic_connection_strengths[synapse] < 0:
                J_ij_i__init += dend_obj.synaptic_connection_strengths[synapse]
        for dendrite in dend_obj.dendritic_inputs:
            if dend_obj.dendritic_connection_strengths[dendrite] >= 0:
                J_ij_e__init += dend_obj.dendritic_connection_strengths[dendrite]
            elif dend_obj.dendritic_connection_strengths[dendrite] < 0:
                if dend_obj.dendritic_inputs[dendrite].name[-15:] != 'dend_refraction': # make sure this isn't the refractory dendrite. that one doesn't get included in this normalization.
                    J_ij_i__init += dend_obj.dendritic_connection_strengths[dendrite]
        if J_ij_e__init > 0:
            factor_e = J_ij_e/J_ij_e__init
        else:
            factor_e = 0
        if J_ij_i__init > 0:
            factor_i = J_ij_i/J_ij_i__init
        else:
            factor_i = 0
        # print('J_ij__init = {}'.format(J_ij__init))
        # print('factor = {}'.format(factor))
        for external_input in dend_obj.external_inputs:
            if dend_obj.external_connection_strengths[external_input] >= 0:
                dend_obj.external_connection_strengths[external_input] = factor_e * dend_obj.external_connection_strengths[external_input]
            elif dend_obj.external_connection_strengths[external_input] < 0:
                dend_obj.external_connection_strengths[external_input] = factor_i * dend_obj.external_connection_strengths[external_input]
        for synapse in dend_obj.synaptic_inputs:
            if dend_obj.synaptic_connection_strengths[synapse] >= 0:
                dend_obj.synaptic_connection_strengths[synapse] = factor_e * dend_obj.synaptic_connection_strengths[synapse]
            elif dend_obj.synaptic_connection_strengths[synapse] < 0:
                dend_obj.synaptic_connection_strengths[synapse] = factor_i * dend_obj.synaptic_connection_strengths[synapse]
        for dendrite in dend_obj.dendritic_inputs:
            if dend_obj.dendritic_connection_strengths[dendrite] >= 0:
                dend_obj.dendritic_connection_strengths[dendrite] = factor_e * dend_obj.dendritic_connection_strengths[dendrite]
            elif dend_obj.dendritic_connection_strengths[dendrite] < 0:
                if dend_obj.dendritic_inputs[dendrite].name[-15:] != 'dend_refraction': # make sure this isn't the refractory dendrite. that one doesn't get included in this normalization.
                    dend_obj.dendritic_connection_strengths[dendrite] = factor_i * dend_obj.dendritic_connection_strengths[dendrite]
        
    # check that timestep is sufficiently small:
    if dend_obj.loops_present == 'ri':
        ib_list = d_params_ri["ib__list"]
        r_fq_array = d_params_ri["r_fq__array"]
    elif dend_obj.loops_present == 'rtti':
        ib_list = d_params_rtti["ib__list"]
        r_fq_array = d_params_rtti["r_fq__array"]
    elif dend_obj.loops_present == 'pri':
        ib_list = d_params_rtti["ib__list"]
        r_fq_array = d_params_rtti["r_fq__array"]


    _ib_ind = index_finder(ib_list,dend_obj.ib)
    _flat_rate = [item for sublist in r_fq_array[_ib_ind] for item in sublist]
    _r_max = np.max(_flat_rate)
    _min = 0.01*dend_obj.beta/dend_obj.alpha
    _max = 0.1*dend_obj.beta/dend_obj.alpha
    if d_tau > 0.1*dend_obj.beta/dend_obj.alpha:
        _str = 'Warning: d_tau should be << beta/alpha.'
        _str = '{} For dendrite {} with beta = {:4.2e} and alpha = {:4.2e}'.format(_str,dend_obj.name,dend_obj.beta,dend_obj.alpha)        
        _str = '{} the recommended d_tau is {:5.3e}-{:5.3e} (dt = {:5.3e}ns-{:5.3e}ns)'.format(_str,_min,_max,_min/t_tau_conversion,_max/t_tau_conversion)
        # print('{}'.format(_str))
    elif d_tau <= 0.1*dend_obj.beta/dend_obj.alpha:
        _str = 'For dendrite {} d_tau = {:5.3e} = {:5.3e} x beta/alpha'.format(dend_obj.name,d_tau,d_tau*dend_obj.alpha/dend_obj.beta)
        _str = '{} (0.01-0.1 x beta/alpha is recommended, dt = {:5.3e}ns-{:5.3e}ns)'.format(_str,_min/t_tau_conversion,_max/t_tau_conversion)
        # print('{}'.format(_str))
    _min = 0.01*dend_obj.beta/_r_max
    _max = 0.1*dend_obj.beta/_r_max
    if d_tau > 0.1*dend_obj.beta/_r_max:
        _str = 'Warning: d_tau should be << beta/r_max.'
        _str = '{} For dendrite {} with beta = {:4.2e} and r_max = {:4.2e}'.format(_str,dend_obj.name,dend_obj.beta,_r_max)        
        _str = '{} the recommended d_tau is {:5.3e}-{:5.3e} (dt = {:5.3e}ns-{:5.3e}ns)'.format(_str,_min,_max,_min/t_tau_conversion,_max/t_tau_conversion)
        # print('{}'.format(_str))
    elif d_tau <= 0.1*dend_obj.beta/_r_max:
        _str = 'For dendrite {} d_tau = {:5.3e} = {:5.3e} x beta/r_max'.format(dend_obj.name,d_tau,d_tau*_r_max/dend_obj.beta)
        _str = '{} (0.01-0.1 x beta/r_max is recommended, dt = {:5.3e}ns-{:5.3e}ns)'.format(_str,_min/t_tau_conversion,_max/t_tau_conversion)
        # print('{}'.format(_str))
    return


def rate_array_attachment(dend_obj):
        
    load_string = f'default_{dend_obj.loops_present}'
        
    ib__list, phi_r__array, i_di__array, r_fq__array, _, _ = dend_load_rate_array(load_string) 
    ib__vec = np.asarray(ib__list)
    
    # attach data to this dendrite

    # bias_current = 2.2 #***
    # _ind__ib = -1 #( np.abs( ib__vec[:] - bias_current ) ).argmin() #***
    # print(ib__vec[-1])
    # _ind__ib = ( np.abs( ib__vec[:] - dend_obj.bias_current ) ).argmin()

    if dend_obj.loops_present == 'pri':
        # print("slide pri")
        _ind__ib = ( np.abs( ib__vec[:] - dend_obj.phi_p ) ).argmin()

    elif dend_obj.loops_present == 'ri':
        _ind__ib = -1
    else:  
        _ind__ib = ( np.abs( ib__vec[:] - dend_obj.bias_current ) ).argmin()
    dend_obj.phi_r__vec = np.asarray(phi_r__array[_ind__ib])
    dend_obj.i_di__subarray = np.asarray(i_di__array[_ind__ib],dtype=object)
    dend_obj.r_fq__subarray = np.asarray(r_fq__array[_ind__ib],dtype=object) 
    
    return

def synapse_initialization(dend_obj,tau_vec,t_tau_conversion):
    
    for synapse_key in dend_obj.synaptic_inputs:
        # print("   Initializing synapse: ", dend_obj.synaptic_inputs[synapse_key].name)
        
        # print('recursive_synapse_initialization:\n  dend_name = {}\n  syn_name = {}\n  in_name = {}\n  spike_times = {}'.format(dend_obj.name,dend_obj.synaptic_inputs[synapse_key].name,dend_obj.synaptic_inputs[synapse_key].input_signal.name,dend_obj.synaptic_inputs[synapse_key].input_signal.spike_times))          
        
        dend_obj.synaptic_inputs[synapse_key]._phi_spd_memory = 0
        dend_obj.synaptic_inputs[synapse_key]._st_ind_last = 0
        dend_obj.synaptic_inputs[synapse_key].phi_spd = np.zeros([len(tau_vec)])
        
        # print('dend = {}, syn = {}'.format(dend_obj.name,dend_obj.synaptic_inputs[synapse_key].name))
        
        if hasattr(dend_obj.synaptic_inputs[synapse_key],'input_signal'):
            if hasattr(dend_obj.synaptic_inputs[synapse_key].input_signal,'input_temporal_form'):
                if dend_obj.synaptic_inputs[synapse_key].input_signal.input_temporal_form == 'constant_rate':
                
                    rate = dend_obj.synaptic_inputs[synapse_key].input_signal.rate * 1e6 # 1e6 because inputs are in MHz
                    # print('rate = {:3.1e}'.format(rate))
                    isi = (1/rate) * 1e9 # 1e9 to convert to ns
                    # print('isi = {:3.1e}'.format(isi))
                    t_f = tau_vec[-1]/t_tau_conversion
                    t_on = dend_obj.synaptic_inputs[synapse_key].input_signal.t_first_spike
                    dend_obj.synaptic_inputs[synapse_key].input_signal.spike_times = np.arange(t_on,t_f+isi,isi)
        else:
            dend_obj.synaptic_inputs[synapse_key].input_signal = dict()
            # dend_obj.synaptic_inputs[synapse_key].input_signal
        # print(dend_obj.synaptic_inputs)#[synapse_key].input_signal,synapse_key)
        dend_obj.synaptic_inputs[synapse_key].spike_times_converted = np.asarray(dend_obj.synaptic_inputs[synapse_key].input_signal.spike_times) * t_tau_conversion
        dend_obj.synaptic_inputs[synapse_key].tau_rise_converted = dend_obj.synaptic_inputs[synapse_key].tau_rise * t_tau_conversion
        dend_obj.synaptic_inputs[synapse_key].tau_fall_converted = dend_obj.synaptic_inputs[synapse_key].tau_fall * t_tau_conversion
        dend_obj.synaptic_inputs[synapse_key].hotspot_duration_converted = dend_obj.synaptic_inputs[synapse_key].hotspot_duration * t_tau_conversion
        dend_obj.synaptic_inputs[synapse_key].spd_duration_converted = dend_obj.synaptic_inputs[synapse_key].spd_duration * t_tau_conversion
        dend_obj.synaptic_inputs[synapse_key].spd_reset_time_converted = dend_obj.synaptic_inputs[synapse_key].spd_reset_time * t_tau_conversion
        
        # remove spike times that came in faster than spd can respond
        if len(dend_obj.synaptic_inputs[synapse_key].spike_times_converted) > 1:
            _spike_times_converted = [dend_obj.synaptic_inputs[synapse_key].spike_times_converted[0]]
            for ii in range(len(dend_obj.synaptic_inputs[synapse_key].spike_times_converted[1:])):
                if dend_obj.synaptic_inputs[synapse_key].spike_times_converted[ii] - _spike_times_converted[-1] >= dend_obj.synaptic_inputs[synapse_key].spd_reset_time_converted:
                    _spike_times_converted.append(dend_obj.synaptic_inputs[synapse_key].spike_times_converted[ii])
            dend_obj.synaptic_inputs[synapse_key].spike_times_converted = _spike_times_converted     
    
    return


def output_synapse_initialization(neuron_object,tau_vec,t_tau_conversion):
    
    for synapse_key in neuron_object.synaptic_outputs:
        
        neuron_object.synaptic_outputs[synapse_key]._phi_spd_memory = 0
        neuron_object.synaptic_outputs[synapse_key]._st_ind_last = 0
        neuron_object.synaptic_outputs[synapse_key].phi_spd = np.zeros([len(tau_vec)])
        
        neuron_object.synaptic_outputs[synapse_key].spike_times_converted = []
        neuron_object.synaptic_outputs[synapse_key].tau_rise_converted = neuron_object.synaptic_outputs[synapse_key].tau_rise * t_tau_conversion
        neuron_object.synaptic_outputs[synapse_key].tau_fall_converted = neuron_object.synaptic_outputs[synapse_key].tau_fall * t_tau_conversion
        neuron_object.synaptic_outputs[synapse_key].hotspot_duration_converted = neuron_object.synaptic_outputs[synapse_key].hotspot_duration * t_tau_conversion
        neuron_object.synaptic_outputs[synapse_key].spd_duration_converted = neuron_object.synaptic_outputs[synapse_key].spd_duration * t_tau_conversion
        neuron_object.synaptic_outputs[synapse_key].spd_reset_time_converted = neuron_object.synaptic_outputs[synapse_key].spd_reset_time * t_tau_conversion
    
    return

def transmitter_initialization(neuron_object,t_tau_conversion):
    
    if neuron_object.source_type == 'qd' or neuron_object.source_type == 'ec':
    
        from soen_utilities import pathfinder
        _path = pathfinder()
        
        if neuron_object.source_type == 'qd':
            load_string = 'source_qd_Nph_1.0e+04'
        elif neuron_object.source_type == 'ec':
            load_string = 'source_ec_Nph_1.0e+04'
            
        with open('{}{}{}.soen'.format(_path,'/soen_sim_data/',load_string), 'rb') as data_file:         
            data_dict_imported = pickle.load(data_file) 
            
        time_vec__el = data_dict_imported['time_vec']#*t_tau_conversion
        el_vec = data_dict_imported['dNphdt']
        t_on_tron = data_dict_imported['t_on_tron']*1e9
        tau_rad = data_dict_imported['tau_rad']
        t_off = np.min([ t_on_tron+5*tau_rad , time_vec__el[-1] ]) 
        
        _ind_on = ( np.abs(t_on_tron-time_vec__el) ).argmin()
        _ind_off = ( np.abs(t_off-time_vec__el) ).argmin()
        
        t_vec__el = time_vec__el[_ind_on:_ind_off] - time_vec__el[_ind_on]
        neuron_object.time_params['tau_vec__electroluminescence'] = t_vec__el * t_tau_conversion
        dt_vec = np.diff(t_vec__el)
        el_vec = el_vec[_ind_on:_ind_off]
    
        # form probability distribution
        el_cumulative_vec = np.cumsum(el_vec[:-1]*dt_vec[:])
        el_cumulative_vec = el_cumulative_vec/np.max(el_cumulative_vec)
        neuron_object.electroluminescence_cumulative_vec = el_cumulative_vec
        
    elif neuron_object.source_type == 'delay_delta':
        
        neuron_object.light_production_delay = neuron_object.light_production_delay * t_tau_conversion 
            
    return

def dendrite_data_attachment(dend_obj,neuron_object):
    
    # attach data to this dendrite
    dend_obj.output_data = {'s': dend_obj.s, 'phi_r': dend_obj.phi_r, 'tau_vec': neuron_object.time_params['tau_vec'], 'time_vec': neuron_object.time_params['tau_vec']/neuron_object.time_params['t_tau_conversion']}
    dend_obj.time_params = {'t_tau_conversion': neuron_object.time_params['t_tau_conversion']}
            
    return        


def construct_dendritic_drives(obj):
                
    for dir_sig in obj.external_inputs:
        
        if hasattr(obj.external_inputs[dir_sig],'piecewise_linear'):
            dendritic_drive = dendritic_drive__piecewise_linear(obj.time_params['time_vec'],obj.external_inputs[dir_sig].piecewise_linear)
                        
        if hasattr(obj.external_inputs[dir_sig],'applied_flux'):
            dendritic_drive = obj.external_inputs[dir_sig].applied_flux * np.ones([len(obj.phi_r_external__vec)])

        # plot_dendritic_drive(time_vec, dendritic_drive)
        obj.external_inputs[dir_sig].drive_signal = dendritic_drive

    return obj


def dendritic_drive__piecewise_linear(time_vec,pwl):
    
    input_signal__dd = np.zeros([len(time_vec)])
    for ii in range(len(pwl)-1):
        t1_ind = (np.abs(np.asarray(time_vec)-pwl[ii][0])).argmin()
        t2_ind = (np.abs(np.asarray(time_vec)-pwl[ii+1][0])).argmin()
        slope = (pwl[ii+1][1]-pwl[ii][1])/(pwl[ii+1][0]-pwl[ii][0])
        partial_time_vec = time_vec[t1_ind:t2_ind+1]
        input_signal__dd[t1_ind] = pwl[ii][1]
        for jj in range(len(partial_time_vec)-1):
            input_signal__dd[t1_ind+jj+1] = input_signal__dd[t1_ind+jj]+(partial_time_vec[jj+1]-partial_time_vec[jj])*slope
    input_signal__dd[t2_ind:] = pwl[-1][1]*np.ones([len(time_vec)-t2_ind])
    
    return input_signal__dd