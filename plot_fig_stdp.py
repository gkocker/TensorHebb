import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import seaborn as sns
from scipy.linalg import norm
from PIL import Image, ImageOps
from inequality.gini import Gini as gini
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import convolve2d
from sklearn import preprocessing
import tensortools as tt
from tensortools import cp_als

from plot_fig_natim import unit_vector, angle_between, get_closest_factor, plot_image_stack, plot_orthog_approx_error
import stdp_pair_params
import stdp_triplet_params

fontsize = 10
labelsize = 8

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ['k']+colors



def generate_inputs(im_dir='/Users/gabeo/Documents/projects/images/berkeley_segmentation/train', xx=10, yy=10):

    im_files = os.listdir(im_dir)
    im_files = [f for f in im_files if f[-3:] == 'jpg']

    K = xx * yy
    inputs = []

    for i, im_file in enumerate(im_files):

        x = Image.open(os.path.join(im_dir, im_file))
        x = ImageOps.grayscale(x)
        x = np.array(x)

        start_x = x.shape[0]//2 + np.random.choice(range(-x.shape[0]//2 + 1, x.shape[0]//2-xx))
        end_x = start_x + xx

        start_y = x.shape[1]//2 + np.random.choice(range(-x.shape[1]//2 + 1, x.shape[1]//2-yy))
        end_y = start_y + yy

        x_new = x[start_x:end_x, start_y:end_y].astype(np.float64)
    #     x_new = x_new - np.amin(x_new)
    #     x_new = x_new / np.amax(x_new)
    #     x_new = preprocessing.scale(x_new)
        # x_new -= np.mean(x_new)
        x_new = preprocessing.normalize(x_new)# - np.mean(x_new), norm='l2')
        inputs.append(x_new)

    inputs = np.array(inputs).astype(np.float32)
    num_inputs = inputs.shape[0]
    inputs = inputs.reshape((num_inputs, K))
    
    return inputs


def phi(g,gain,p=1):
    
    import numpy as np
    ind = np.where(g<0)
    g[ind[0]] = 0

    r_out = gain*(g**p)

    return(r_out)


def g_fun(s, tau=10.):
    return s * np.exp(-np.abs(s) / tau) / (tau**2) 


def g_fun_laplace(w, tau=10.): ### alpha function laplace transform, no delay
    return 1 / (1 + w*tau)**2


def sim_poisson_pairSTDP_record_inputs(inputs, par, tstop=10000, trans=100, dt=.01, Nt_downsample=200, J0=None, tswitch=100):

    num_inputs, K = inputs.shape
    
    # unpackage parameters
    Ne = par.Ne
    N = par.N
    tau = par.tau
    b = par.b
    gain = par.gain
    A3plus = par.A3plus
    A2minus = par.A2minus
    A2plus = par.A2plus
    tauplus = par.tauplus
    tauminus = par.tauminus
    taux = par.taux
    tauy = par.tauy
    eta = par.eta
    p = par.p
    pp = par.p_plast

    A3plus = eta * A3plus
    A2minus = eta * A2minus
    A2plus = eta * A2plus

    ### pre- and postsynaptic plasticity traces
    r1 = np.zeros((K,)) # for each input
    o1 = np.zeros((N,)) # for each postsynaptic cell

    Nt = int(tstop / dt)
    Ntrans = int(trans / dt)
    t = 0
    numspikes = 0

    count = np.zeros((N,))  # spike count of each neuron in the interval (0,tstop)
    maxspikes = int(100 * K * tstop / 1000)  # 100 Hz / neuron
    spktimes = np.zeros((maxspikes, 2))  # spike times and neuron labels
    
    #    Nt_downsample = 100
    dt_downsample = (tstop-trans) / Nt_downsample
    Jt = np.zeros((Nt_downsample, Ne, K))
    trace_t = np.zeros((Nt,))

    s = np.zeros((K,))  # synaptic outputs
    s_dummy = np.zeros((K,))
    spiket = np.zeros((N,)) # spiking outputs

    a = 1. / tau
    a2 = a ** 2

    if J0 is None:
        J = np.random.randn(Ne,K)
        J /= np.linalg.norm(J, ord=p, axis=1)
    else:
        J = J0

    x = inputs[np.random.choice(num_inputs)]      

    for i in range(0, Nt, 1):

        t += dt

        if np.fmod(i-Ntrans, tswitch/dt) == 0: # switch input
            x = inputs[np.random.choice(num_inputs)]      
        else: pass

        try:
            pre_spk = np.random.poisson(x*dt, size=(K,))
        except:
            break
        
        # update the output and plasticity traces
        s_dummy += dt * (-2 * a * s_dummy - a2 * s)
        s += dt * (s_dummy + a2*pre_spk) # a2 for normalization of integral

        r1 += dt * (-1. / tauplus * r1)
        o1 += dt * (-1. / tauminus * o1)

        # compute input
        g = J.dot(s) + b

        # decide if each neuron spikes, update synaptic output of spiking neurons
        # each neurons's rate is phi(g)
        r = phi(g, gain=gain, p=p)

        try:
            spiket = np.random.poisson(r * dt, size=(1,))
        except:
            break

        ### store spike times and counts
        if t > trans:
            count += spiket
            for j in range(K):
                if (pre_spk[j] >= 1) and (numspikes < maxspikes):
                    spktimes[numspikes, 0] = t-trans
                    spktimes[numspikes, 1] = j
                    numspikes += 1

            ### store weight matrix as a function of time
            if np.fmod(i - Ntrans, ((Nt - Ntrans) / Nt_downsample)) == 0:
                Jt[int((t-trans) / dt_downsample)] = J

        ### plasticity
        o1 += spiket[0:Ne]
        r1 += spiket[0:Ne]
        
        if t > trans:
            dJ = np.zeros((Ne, K))
            
            # presynaptic spikes cause depression proportional to postsynaptic traces
            ind_spk = np.where(pre_spk > 0)[0]       
            for j in ind_spk:
                dJ[:, j] -= A2minus * o1 

            # postsynaptic spikes cause potentiation prop. to presynaptic trace
            ind_spk = np.where(spiket > 0)[0]
            for j in ind_spk:
                dJ[j, :] += A3plus * r1 
    
            J += dt * (dJ - J * np.sum(J  * dJ * np.abs(J)**(pp-2), axis=1)) 


    # truncate spike time array
    spktimes = spktimes[0:numspikes, :]

    return spktimes, Jt


def sim_poisson_tripletSTDP_record_inputs(inputs, par, tstop=10000, trans=100, dt=.01, Nt_downsample=200, J0=None, tswitch=100):

    num_inputs, K = inputs.shape
    
    # unpackage parameters
    Ne = par.Ne
    N = par.N
    tau = par.tau
    b = par.b
    gain = par.gain
    A3plus = par.A3plus
    A2minus = par.A2minus
    A2plus = par.A2plus
    tauplus = par.tauplus
    tauminus = par.tauminus
    taux = par.taux
    tauy = par.tauy
    eta = par.eta
    p = par.p
    pp = par.p_plast

    A3plus = eta * A3plus
    A2minus = eta * A2minus
    A2plus = eta * A2plus

    ### pre- and postsynaptic plasticity traces
    r1 = np.zeros((K,)) # for each input
    o1 = np.zeros((N,)) # for each postsynaptic cell
    o2 = np.zeros((K,))

    Nt = int(tstop / dt)
    Ntrans = int(trans / dt)
    t = 0
    numspikes = 0

    count = np.zeros((N,))  # spike count of each neuron in the interval (0,tstop)
    maxspikes = int(1000 * K * tstop / 1000)  # 200 Hz / neuron
    spktimes = np.zeros((maxspikes, 2))  # spike times and neuron labels
    
    #    Nt_downsample = 100
    dt_downsample = (tstop-trans) / Nt_downsample
    Jt = np.zeros((Nt_downsample, Ne, K))
    trace_t = np.zeros((Nt,))

    s = np.zeros((K,))  # synaptic outputs
    s_dummy = np.zeros((K,))
    spiket = np.zeros((N,)) # spiking outputs

    a = 1. / tau
    a2 = a ** 2

    if J0 is None:
        J = np.random.randn(Ne,K)
        J /= np.linalg.norm(J, ord=p, axis=1)
    else:
        J = J0
    
    x = inputs[np.random.choice(num_inputs)]  

    for i in range(0, Nt, 1):

        t += dt

        # pick input

        if np.fmod(i-Ntrans, tswitch/dt) == 0: # switch input
            x = inputs[np.random.choice(num_inputs)]      
        else: pass

        try:
            pre_spk = np.random.poisson(x*dt, size=(K,))
        except:
            break
        
        # update the output and plasticity traces
        s_dummy += dt * (-2 * a * s_dummy - a2 * s)
        s += dt * (s_dummy + a2*pre_spk) # a2 for normalization of integral

        r1 += dt * (-1. / tauplus * r1)
        o1 += dt * (-1. / tauminus * o1)
        o2 += dt * (-1. / tauy * o2)

        # compute input
        g = J.dot(s) + b

        # decide if each neuron spikes, update synaptic output of spiking neurons
        # each neurons's rate is phi(g)
        r = phi(g, gain=gain, p=p)

        try:
            spiket = np.random.poisson(r * dt, size=(1,))
        except:
            break

        ### store spike times and counts
        if t > trans:

            if spiket >= 1:
                spktimes[numspikes, 0] = t-trans
                spktimes[numspikes, 1] = K
                numspikes += 1

            count += spiket
            for j in range(K):
                if (pre_spk[j] >= 1) and (numspikes < maxspikes):
                    spktimes[numspikes, 0] = t-trans
                    spktimes[numspikes, 1] = j
                    numspikes += 1

            ### store weight matrix as a function of time
            if np.fmod(i - Ntrans, ((Nt - Ntrans) / Nt_downsample)) == 0:
                Jt[int((t-trans) / dt_downsample)] = J

        ### plasticity
        o1 += spiket[0:Ne]
        r1 += spiket[0:Ne]
        
        if t > trans:
            dJ = np.zeros((Ne, K))
            ind_spk = np.where(pre_spk > 0)[0]
                        
            for j in ind_spk:
                dJ[:, j] -= A2minus * o1 # presynaptic spikes cause depression proportional to postsynaptic traces

            ind_spk = np.where(spiket > 0)[0]
            for j in ind_spk:
                dJ[j, :] += A3plus * r1 * o2 # postsynaptic spikes cause potentiation prop. to presynaptic traces
    
            J += dt * (dJ - J * np.sum(J  * dJ * np.abs(J)**(pp-2), axis=1)) 

        o2 += spiket[0:Ne]

    # truncate spike time array
    spktimes = spktimes[0:numspikes, :]

    return spktimes, Jt


def sim_poisson_pairSTDP(inputs, par, tstop=10000, trans=100, dt=.01, Nt_downsample=200, J0=None, tswitch=100):

    num_inputs, K = inputs.shape
    
    # unpackage parameters
    Ne = par.Ne
    N = par.N
    tau = par.tau
    b = par.b
    gain = par.gain
    A3plus = par.A3plus
    A2minus = par.A2minus
    A2plus = par.A2plus
    tauplus = par.tauplus
    tauminus = par.tauminus
    taux = par.taux
    tauy = par.tauy
    eta = par.eta
    p = par.p
    pp = par.p_plast

    A3plus = eta * A3plus
    A2minus = eta * A2minus
    A2plus = eta * A2plus

    ### pre- and postsynaptic plasticity traces
    r1 = np.zeros((K,)) # for each input
    o1 = np.zeros((N,)) # for each postsynaptic cell

    Nt = int(tstop / dt)
    Ntrans = int(trans / dt)
    t = 0
    numspikes = 0

    count = np.zeros((N,))  # spike count of each neuron in the interval (0,tstop)
    maxspikes = int(100 * N * tstop / 1000)  # 100 Hz / neuron
    spktimes = np.zeros((maxspikes, 2))  # spike times and neuron labels
    
    #    Nt_downsample = 100
    dt_downsample = (tstop-trans) / Nt_downsample
    Jt = np.zeros((Nt_downsample, Ne, K))
    trace_t = np.zeros((Nt,))

    s = np.zeros((K,))  # synaptic outputs
    s_dummy = np.zeros((K,))
    spiket = np.zeros((N,)) # spiking outputs

    a = 1. / tau
    a2 = a ** 2

    if J0 is None:
        J = np.random.randn(Ne,K)
        J /= np.linalg.norm(J, ord=p, axis=1)
    else:
        J = J0

    x = inputs[np.random.choice(num_inputs)]      

    for i in range(0, Nt, 1):

        t += dt

        if np.fmod(i-Ntrans, tswitch/dt) == 0: # switch input
            x = inputs[np.random.choice(num_inputs)]      
        else: pass

        try:
            pre_spk = np.random.poisson(x*dt, size=(K,))
        except:
            break
        
        # update the output and plasticity traces
        s_dummy += dt * (-2 * a * s_dummy - a2 * s)
        s += dt * (s_dummy + a2*pre_spk) # a2 for normalization of integral

        r1 += dt * (-1. / tauplus * r1)
        o1 += dt * (-1. / tauminus * o1)

        # compute input
        g = J.dot(s) + b

        # decide if each neuron spikes, update synaptic output of spiking neurons
        # each neurons's rate is phi(g)
        r = phi(g, gain=gain, p=p)

        try:
            spiket = np.random.poisson(r * dt, size=(1,))
        except:
            break

        ### store spike times and counts
        if t > trans:
            count += spiket
            for j in range(N):
                if (spiket[j] >= 1) and (numspikes < maxspikes):
                    spktimes[numspikes, 0] = t
                    spktimes[numspikes, 1] = j
                    numspikes += 1

            ### store weight matrix as a function of time
            if np.fmod(i - Ntrans, ((Nt - Ntrans) / Nt_downsample)) == 0:
                Jt[int((t-trans) / dt_downsample)] = J

        ### plasticity
        o1 += spiket[0:Ne]
        r1 += spiket[0:Ne]
        
        if t > trans:
            dJ = np.zeros((Ne, K))
            
            # presynaptic spikes cause depression proportional to postsynaptic traces
            ind_spk = np.where(pre_spk > 0)[0]       
            for j in ind_spk:
                dJ[:, j] -= A2minus * o1 

            # postsynaptic spikes cause potentiation prop. to presynaptic trace
            ind_spk = np.where(spiket > 0)[0]
            for j in ind_spk:
                dJ[j, :] += A3plus * r1 
    
            J += dt * (dJ - J * np.sum(J  * dJ * np.abs(J)**(pp-2), axis=1)) 


    # truncate spike time array
    spktimes = spktimes[0:numspikes, :]

    return spktimes, Jt


def sim_poisson_tripletSTDP(inputs, par, tstop=10000, trans=100, dt=.01, Nt_downsample=200, J0=None, tswitch=100):

    num_inputs, K = inputs.shape
    
    # unpackage parameters
    Ne = par.Ne
    N = par.N
    tau = par.tau
    b = par.b
    gain = par.gain
    A3plus = par.A3plus
    A2minus = par.A2minus
    A2plus = par.A2plus
    tauplus = par.tauplus
    tauminus = par.tauminus
    taux = par.taux
    tauy = par.tauy
    eta = par.eta
    p = par.p
    pp = par.p_plast

    A3plus = eta * A3plus
    A2minus = eta * A2minus
    A2plus = eta * A2plus

    ### pre- and postsynaptic plasticity traces
    r1 = np.zeros((K,)) # for each input
    o1 = np.zeros((N,)) # for each postsynaptic cell
    o2 = np.zeros((K,))

    Nt = int(tstop / dt)
    Ntrans = int(trans / dt)
    t = 0
    numspikes = 0

    count = np.zeros((N,))  # spike count of each neuron in the interval (0,tstop)
    maxspikes = int(100 * N * tstop / 1000)  # 100 Hz / neuron
    spktimes = np.zeros((maxspikes, 2))  # spike times and neuron labels
    
    #    Nt_downsample = 100
    dt_downsample = (tstop-trans) / Nt_downsample
    Jt = np.zeros((Nt_downsample, Ne, K))
    trace_t = np.zeros((Nt,))

    s = np.zeros((K,))  # synaptic outputs
    s_dummy = np.zeros((K,))
    spiket = np.zeros((N,)) # spiking outputs

    a = 1. / tau
    a2 = a ** 2

    if J0 is None:
        J = np.random.randn(Ne,K)
        J /= np.linalg.norm(J, ord=p, axis=1)
    else:
        J = J0
    
    x = inputs[np.random.choice(num_inputs)]  

    for i in range(0, Nt, 1):

        t += dt

        # pick input

        if np.fmod(i-Ntrans, tswitch/dt) == 0: # switch input
            x = inputs[np.random.choice(num_inputs)]      
        else: pass

        try:
            pre_spk = np.random.poisson(x*dt, size=(K,))
        except:
            break
        
        # update the output and plasticity traces
        s_dummy += dt * (-2 * a * s_dummy - a2 * s)
        s += dt * (s_dummy + a2*pre_spk) # a2 for normalization of integral

        r1 += dt * (-1. / tauplus * r1)
        o1 += dt * (-1. / tauminus * o1)
        o2 += dt * (-1. / tauy * o2)

        # compute input
        g = J.dot(s) + b

        # decide if each neuron spikes, update synaptic output of spiking neurons
        # each neurons's rate is phi(g)
        r = phi(g, gain=gain, p=p)

        try:
            spiket = np.random.poisson(r * dt, size=(1,))
        except:
            break

        ### store spike times and counts
        if t > trans:
            count += spiket
            for j in range(N):
                if (spiket[j] >= 1) and (numspikes < maxspikes):
                    spktimes[numspikes, 0] = t
                    spktimes[numspikes, 1] = j
                    numspikes += 1

            ### store weight matrix as a function of time
            if np.fmod(i - Ntrans, ((Nt - Ntrans) / Nt_downsample)) == 0:
                Jt[int((t-trans) / dt_downsample)] = J

        ### plasticity
        o1 += spiket[0:Ne]
        r1 += spiket[0:Ne]
        
        if t > trans:
            dJ = np.zeros((Ne, K))
            ind_spk = np.where(pre_spk > 0)[0]
                        
            for j in ind_spk:
                dJ[:, j] -= A2minus * o1 # presynaptic spikes cause depression proportional to postsynaptic traces

            ind_spk = np.where(spiket > 0)[0]
            for j in ind_spk:
                dJ[j, :] += A3plus * r1 * o2 # postsynaptic spikes cause potentiation prop. to presynaptic traces
    
            J += dt * (dJ - J * np.sum(J  * dJ * np.abs(J)**(pp-2), axis=1)) 

        o2 += spiket[0:Ne]

    # truncate spike time array
    spktimes = spktimes[0:numspikes, :]

    return spktimes, Jt


def loop_initial_cond(par, input_scale=0.1, tstop=100000, dt=.1, Nt_downsample=1000, Nsave=100, Ninit=20, Npix=10, datafile_head='_data_orth_error', rerun=True, stdp='pair'):

    datafile = 'stdp_{}_d={}'.format(stdp, par.p)+datafile_head+'.pkl'

    if os.path.exists(datafile):
        print('Model ensemble already exists at {}'.format(datafile))
        
        if not rerun:
            with open(datafile, 'rb') as handle:
                results_dict = pickle.load(handle)
            return results_dict['angle_factor_weights']

        else: os.remove(datafile)

    else: pass

    if Nsave > Npix**2:
        Nsave = Npix**2

    theta = np.zeros((Nt_downsample, Ninit))
    factors = np.zeros((Npix**2, Ninit))
    Jsave = np.zeros((Nt_downsample, Nsave, Ninit))
    indices = np.zeros((Nsave, Ninit))

    results_dict = {}
    results_dict['Npix'] = Npix
    results_dict['parameters'] = par

    for n in range(Ninit):

        inputs = generate_inputs(xx=Npix, yy=Npix)
        inputs *= input_scale
        num_inputs, K = inputs.shape

        if stdp == 'pair':

            _, Jt = sim_poisson_pairSTDP(inputs, par, tstop=tstop, trans=100, dt=dt, Nt_downsample=Nt_downsample, J0=None)
            Jt = np.squeeze(Jt)

            if par.p == 1:
                mu = np.einsum('ij,ik',inputs,inputs) / num_inputs 
                mu *= par.A2minus*g_fun_laplace(1/par.tauminus) + par.A2plus*g_fun_laplace(1/par.tauplus)

                ### convolution with synaptic filters and stdp kernel. 
                ### synpatic kernel promotes mu to a function of time lags.
                ### integration against each component of the stdp kernel is a laplace transform. it picks out one frequency for each time lag.

                lam, v = np.linalg.eig(mu)
                v = v.T
                factor = v[np.argmax(lam)]

            elif par.p == 2:
                mu = np.einsum('ij,ik,il', inputs, inputs, inputs) / num_inputs
                mu *= par.A2minus*g_fun_laplace(1/par.tauminus) + par.A2plus*g_fun_laplace(1/par.tauplus)

                U = tt.cp_als(mu, rank=1, verbose=False)
                factor = get_closest_factor(U, Jt[-10:].mean(axis=0), a=2)

            else:
                raise Exception('write correlation for pair stdp with d={}'.format(par.p))

            if angle_between(Jt[-1], factor) > np.pi/2:
                factor *= -1
            
            for t in range(Nt_downsample):
                theta[t, n] = angle_between(Jt[t], factor)

            factors[:, n] = factor
            del mu
            del Jt

        elif stdp == 'triplet':
            _, Jt = sim_poisson_tripletSTDP(inputs, par, tstop=tstop, trans=100, dt=dt, Nt_downsample=Nt_downsample, J0=None)
            Jt = np.squeeze(Jt)

            a = [1, 2]
            ahat = max(a)
            N = len(a)

            if par.p == 1:

                M = np.zeros([K for m in range(ahat+1)])

                for m in range(N):
                    if a[m] == 1:
                        mu = np.einsum('ij,ik',inputs,inputs) / num_inputs
                        mu *= par.A2minus * g_fun_laplace(1/par.tauminus) + par.A2plus * g_fun_laplace(1/par.tauplus)
                        M[:, :, 0] += mu
                    elif a[m] == 2:
                        mu = np.einsum('ij,ik,il',inputs,inputs,inputs) / num_inputs
                        mu *= par.A3plus * g_fun_laplace(1/par.tauplus) * g_fun_laplace(1/par.tauy)
                        M[:, :, :] += mu
                    else:
                        raise Exception('write input correlation computation for m={}'.format(m))

                U = tt.cp_als(M, rank=1, verbose=False)
                factor = get_closest_factor(U, Jt[-10:].mean(axis=0), a=ahat)

            elif par.p == 2: 
                
                M = np.zeros([K for m in range(ahat+par.p+1)])
                
                for m in range(N):
                    if a[m] == 1:
                        mu = np.einsum('ij,ik',inputs,inputs) / num_inputs
                        mu *= par.A2minus * g_fun_laplace(1/par.tauminus) + par.A2plus * g_fun_laplace(1/par.tauplus)
                        M[:, :, 0] += mu

                    elif a[m] == 2:

                        mu = np.einsum('ij,ik,il', inputs, inputs, inputs) / num_inputs #line 1
                        mu *= par.A2minus * g_fun_laplace(1/par.tauminus) + par.A2plus * g_fun_laplace(1/par.tauplus)
                        mu *= par.A3plus * g_fun(0.)

                        M[:, :, :, 0, 0] += mu 

                        mu = np.einsum('ij, il', inputs, inputs) / num_inputs #line 2, x^d
                        M[:, :, 0, 0, 0] += mu

                        mu = np.einsum('ij, ik, il', inputs, inputs, inputs) / num_inputs #line 2, x^d x_i
                        mu *= par.A2minus * g_fun_laplace(1/par.tauminus) + par.A2plus * g_fun_laplace(1/par.tauplus)
                        mu *= par.A3plus * g_fun_laplace(1/par.tauy)

                        M[0, 0, :, :, :] += mu

                        mu = np.einsum('ij, ik', inputs, inputs) / num_inputs # line 3
                        mu *= par.A3plus * g_fun_laplace(1/par.tauy)

                        M[:, :, 0, 0, 0] += mu

                        mu = np.einsum('ij, ik, il', inputs, inputs, inputs) / num_inputs
                        mu -= np.outer(np.einsum('ij, ik', inputs, inputs), np.sum(inputs, axis=0)) / (num_inputs**2)
                        mu *= par.A2minus * g_fun_laplace(1/par.tauminus) + par.A2plus * g_fun_laplace(1/par.tauplus)
                        M[0, 0, :, :, :] += mu

                U = tt.cp_als(M, rank=1, verbose=False)
                factor = get_closest_factor(U, Jt[-10:].mean(axis=0), a=ahat+par.p+1)

            else:
                raise Exception('write correlation for pair stdp with d={}'.format(par.p))

            for t in range(Nt_downsample):
                theta[t, n] = angle_between(Jt[t], factor)

            factors[:, n] = factor

            ind_save = np.random.choice(Npix**2, size=Nsave, replace=False)
            indices[:, n] = ind_save
            Jsave[:, :, n] = Jt[:, ind_save]

            del mu
            del M
            del Jt

        else:
            raise Exception('Unknown stdp kernel: {}'.format(stdp))
    
    results_dict['factors'] = factors
    results_dict['angle_factor_weights'] = theta
    results_dict['weights_over_time'] = Jsave
    results_dict['weight_indices'] = indices

    with open(datafile, 'wb') as handle:
        pickle.dump(results_dict, handle)

    return theta


def plot_weighted_combined_3pt_corr(ax, inputs, par, a=[1, 2], b=[1, 1], Nplot=4, Nsmooth=31, cmap='cividis', cscale=.35):

    N = len(a)
    num_inputs, K = inputs.shape

    plot_ind = list(range(0, K, K//(Nplot)))
    if len(plot_ind) == Nplot+1:
        plot_ind = plot_ind[:-1]

    if K % Nsmooth == 0:
        Kplot = K//(Nsmooth//2)
    else:
        Kplot = K//(Nsmooth//2) + 1

    Xmesh, Ymesh = np.meshgrid(np.arange(Kplot), np.arange(Kplot))

    ahat = max(a)
    M = np.zeros([Kplot for m in range(ahat)]+[Nplot])

    for m in range(N):
        inputsb = inputs**b[m]

        if a[m] == 1:
            mui = np.einsum('ij,ik',inputs,inputs) / num_inputs
            mui = convolve2d(mui, np.ones((Nsmooth, Nsmooth)), mode='same', boundary='wrap')
            mui = mui[::Nsmooth//2, ::Nsmooth//2]
            mui *= par.A2minus * g_fun_laplace(1/par.tauminus) + par.A2plus * g_fun_laplace(1/par.tauplus)

            M[:, :, 0] += mui

        elif a[m] == 2:
            for i, z in enumerate(plot_ind):
                inputs2 = inputs * np.outer(inputs[:, z], np.ones(K,))
                mui = inputsb.T.dot(inputs2) / num_inputs
                mui = convolve2d(mui, np.ones((Nsmooth, Nsmooth)), mode='same', boundary='wrap')
                mui = mui[::Nsmooth//2, ::Nsmooth//2]
                mui *= par.A3plus * g_fun_laplace(1/par.tauplus) * g_fun_laplace(1/par.tauy)

                M[:, :, i] += mui

        else:
            print('write input correlation computation for m={}, a[m]={}'.format(m, a[m]))

    del mui

    for z in range(Nplot):
        Zmesh = z * np.ones((Kplot, Kplot))
        
        Mz = M[:, :, z]
        cmid = np.mean(Mz)
        cmin = cmid - (np.amax(Mz) - np.amin(Mz))*cscale
        cmax = cmid + (np.amax(Mz) - np.amin(Mz))*cscale

        scam = plt.cm.ScalarMappable(
            norm=plt.cm.colors.Normalize(cmin, cmax),
            cmap=cmap
        )

        scam.set_array([])   
        surf = ax.plot_surface(
            Xmesh, Ymesh, Zmesh,
            facecolors  = scam.to_rgba(Mz),
            antialiased = True,
            rstride=1, cstride=1, alpha=None
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    ax.set_xlabel('Input', fontsize=fontsize)
    ax.set_ylabel('Input', fontsize=fontsize)
    ax.set_zlabel('Input', fontsize=fontsize)

    return scam, plot_ind


def plot_fig_pair_stdp(savefile='fig_pairstdp.pdf', input_scale=0.1, Npix=10):


    par = stdp_pair_params.params()

    fig, ax = plt.subplots(2, 3, figsize=(5.67, 3.7))

    s = np.arange(-100, 100, .1)
    A = np.zeros(len(s))

    ind = np.where(s<0)[0]
    A[ind] = -par.A2minus * np.exp(s[ind]/par.tauminus)
    
    ind = np.where(s>0)[0]
    A[ind] = par.A2plus * np.exp(-s[ind]/par.tauplus)

    ax[0, 0].plot(s, A, 'k', linewidth=2)
    ax[0, 0].set_xlabel('Pre-post lag (ms)')
    ax[0, 0].set_ylabel('STDP kernel')

    x = np.arange(-2, 2, .01)
    y = np.zeros((len(x)))
    theta = x>0

    ax[1, 0].plot(x, theta * x, linewidth=2, label='linear', color=colors[0])
    ax[1, 0].plot(x, theta * x**2, linewidth=2, label='quadratic', color=colors[1])

    ax[1, 0].set_xlabel('Potential (mV)')
    ax[1, 0].set_ylabel('Rate (sp/ms)')

    ### pair stdp, linear neuron
    par.b = .05 * np.ones((par.N,))

    inputs = generate_inputs(xx=Npix, yy=Npix)
    num_inputs, K = inputs.shape
    inputs *= input_scale

    tstop = 400
    trans = 100
    tswitch = 100
    num_plot = 1
    Nt_ds = 200

    plotind = np.random.choice(K, size=num_plot, replace=False)
    spktimes, Jt = sim_poisson_pairSTDP_record_inputs(inputs, par, tstop=tstop+trans, trans=trans, tswitch=tswitch, Nt_downsample=Nt_ds)
    Jt = np.squeeze(Jt)

    tplot = np.arange(0, tstop, tstop/Nt_ds)

    ax[0, 1].plot(spktimes[:, 0], spktimes[:, 1], 'k|', markersize=1)

    for i in range(tstop//tswitch):
        x = [tswitch*i, tswitch*i]
        y = [0, K]
        ax[0, 1].plot(x, y, 'k')

    ax[1, 1].plot(tplot, Jt[:, plotind], linewidth=2)

    for i in range(2): ax[i, 1].set_xlim((0, tstop))
    
    ax[1, 1].set_xlabel('Time (ms)', fontsize=fontsize)
    ax[0, 1].set_ylabel('Input', fontsize=fontsize)
    ax[1, 1].set_ylabel('Synaptic weight (mV)', fontsize=fontsize)

    tstop = 200000
    spktimes, Jt = sim_poisson_pairSTDP(inputs, par, tstop=tstop, tswitch=tswitch, Nt_downsample=Nt_ds)
    Jt = np.squeeze(Jt)

    plot_ind = np.random.choice(K, 5)
    tplot = np.arange(0, tstop, tstop/Nt_ds) / 1000
    ax[0, 2].plot(tplot, Jt[:, plot_ind])

    ax[1, 2].set_xlabel('Time (min)')
    ax[0, 2].set_ylabel('Synaptic weight (mV)')

    ### pair stdp, quadratic neuron
    par = stdp_pair_params.params()
    par.b = .05 * np.ones((par.N,))
    tstop = 200000


    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(savefile)


def plot_fig_triplet_stdp(Npix=10, savefile='fig_tripletstdp.pdf', cmap='cividis', cscale=.35, Nsmooth=31):


    fig = plt.figure(figsize=(4.75, 6.25))

    gs0 = gs.GridSpec(5, 1, figure=fig)

    gs00 = gs.GridSpecFromSubplotSpec(2, 4, subplot_spec=gs0[:2]) # triplet stdp rule and the input correlation it "decomposes"
    ax1 = fig.add_subplot(gs00[0, 0], projection='3d')
    ax2 = fig.add_subplot(gs00[1, 0], projection='3d')

    gs01 = gs.GridSpecFromSubplotSpec(10, 1, subplot_spec=gs00[:, 1:]) # short sim example with rasters
    # ax3 = fig.add_subplot(gs01[1]) 
    # ax4 = fig.add_subplot(gs01[2:7])
    ax5 = fig.add_subplot(gs01[8:])


    ax6 = fig.add_subplot(gs0[2]) # long sim example

    gs02 = gs.GridSpecFromSubplotSpec(1, 3, subplot_spec=gs0[3]) # pair stdp curve, its input corr, firing rate functions 
    ax7 = fig.add_subplot(gs02[0])
    ax8 = fig.add_subplot(gs02[1])
    ax9 = fig.add_subplot(gs02[2])

    ax10 = fig.add_subplot(gs0[-1]) # angle between first factor and weight vector

    par = stdp_triplet_params.params()
    par.b = .015 * np.ones((par.N,))

    t_th = np.arange(-150.,150.,1)
    Nt_calc = t_th.size
    ind0 = Nt_calc//2

    L = np.zeros(Nt_calc)
    L[:ind0] = -par.A2minus*np.exp(-np.abs(t_th[:ind0])/par.tauminus)

    Q = np.zeros((Nt_calc, Nt_calc))
    Q[ind0:,ind0+1:] = par.A3plus*np.outer(np.exp(-np.abs(t_th[ind0:])/par.tauplus), np.exp(-np.abs(t_th[ind0+1:])/par.tauy))

    ax1.plot(xs=t_th, ys=t_th[ind0]*np.ones(Nt_calc), zs=Q[:, ind0]+L, color='k')
    ax1.plot(xs=t_th, ys=t_th[ind0+50]*np.ones(Nt_calc), zs=Q[:, ind0+50]+L, color='k')
    ax1.plot(xs=t_th, ys=t_th[ind0+100]*np.ones(Nt_calc), zs=Q[:, ind0+100]+L, color='k')
    ax1.set_xlabel('Post-pre\nlag (ms)', fontsize=fontsize)
    ax1.set_ylabel('Post-post\nlag (ms)', fontsize=fontsize)
    ax1.set_zlabel('Synaptic weight\nchange (mV)', fontsize=fontsize)

    ax1.view_init(elev=10)
    ax1.set_zticks([])
    ax1.set_xticks([t_th[ind0], t_th[ind0-100], t_th[ind0+100]])
    ax1.set_yticks([t_th[ind0], t_th[ind0-100], t_th[ind0+100]])

    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    ax1.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax1.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax1.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

    inputs = generate_inputs(xx=Npix, yy=Npix)
    K, num_inputs = inputs.shape
    _, _ = plot_weighted_combined_3pt_corr(ax2, inputs, par, cmap=cmap, cscale=cscale, Nsmooth=Nsmooth)

    ### pair depression, triplet potentiation, linear neuron: long and short sim examples
    par = stdp_triplet_params.params()
    par.b = .015 * np.ones((par.N,))
    tstop = 400
    trans = 100
    tstop += trans

    dt = .01
    Nt_ds = 200

    K = Npix**2
    spktimes, Jt = sim_poisson_tripletSTDP_record_inputs(inputs, par, tstop=tstop, trans=trans, dt=dt, Nt_downsample=Nt_ds, J0=None, tswitch=100)

    # output_spktimes = spktimes[np.where(spktimes[:, 1] == K)[0], 0]
    # ax3.plot(output_spktimes, np.ones(len(output_spktimes)), 'k|')
    # ax3.set_yticks([])

    # input_spktimes = spktimes[np.where(spktimes[:, 1] < K)[0], :]
    # spkind = range(0, len(input_spktimes), 10)

    # ax4.plot(input_spktimes[spkind, 0], input_spktimes[spkind, 1], 'k|', markersize=0.5)
    # ax4.set_ylabel('Input', fontsize=fontsize)

    # ax3.set_frame_on(False)
    # ax3.set_xticks([])
    # ax4.set_ylim((0, K))
    # ax3.set_xlim((0, tstop-100))
    # ax4.set_xlim((0, tstop-100))

    tplot = np.linspace(0, tstop-100, Nt_ds)
    plotind = np.random.choice(K, size=20, replace=False)
    ax5.plot(tplot, Jt[:, 0, plotind], 'k', linewidth=.5)

    ax5.set_xlim((0, tstop-100))

    ax5.set_ylabel('Synaptic\nweight (mV)', fontsize=fontsize)
    ax5.set_xlabel('Time (ms)', fontsize=fontsize)

    tstop = 5000
    trans = 100
    tstop += trans
    _, Jt = sim_poisson_tripletSTDP_record_inputs(inputs, par, tstop=tstop, trans=trans, dt=dt, Nt_downsample=Nt_ds, J0=None, tswitch=100)

    tplot = np.linspace(0, tstop-trans, Nt_ds) / 1000
    plotind = np.random.choice(K, size=20, replace=False)
    ax6.plot(tplot, Jt[:, 0, plotind], 'k', linewidth=.5)
    ax6.set_xlim((0, tplot[-1]))
    ax6.set_xlabel('Time (s)', fontsize=fontsize)
    ax6.set_ylabel('Synaptic\nweight (mV)', fontsize=fontsize)

    # pair stdp curve
    par = stdp_pair_params.params()

    t_th = np.arange(-150.,150.,1)
    Nt_calc = t_th.size
    ind0 = Nt_calc//2

    L = np.zeros(Nt_calc)
    L[:ind0] = -par.A2minus*np.exp(-np.abs(t_th[:ind0])/par.tauminus)
    L[ind0:] = par.A2plus * np.exp(-np.abs(t_th[ind0:])/par.tauplus)

    ax7.plot(t_th, L, color=colors[1], linewidth=2)
    ax7.set_xlabel('Pre-post lag (ms)', fontsize=fontsize)
    ax7.set_ylabel('Synaptic weight\nchange (mV)', fontsize=fontsize)


    mui = np.einsum('ij,ik',inputs,inputs) / num_inputs
    mui = convolve2d(mui, np.ones((Nsmooth, Nsmooth)), mode='same', boundary='wrap')
    mui = mui[::Nsmooth//2, ::Nsmooth//2]
    mui *= par.A2minus * g_fun_laplace(1/par.tauminus) + par.A2plus * g_fun_laplace(1/par.tauplus)

    cmid = np.mean(mui)
    cmin = cmid - (np.amax(mui) - np.amin(mui))*cscale
    cmax = cmid + (np.amax(mui) - np.amin(mui))*cscale

    ax8.imshow(mui, cmap=cmap, clim=(cmin, cmax))
    ax8.set_xticks([])
    ax8.set_yticks([])
    ax8.set_xlabel('Input', fontsize=fontsize)
    ax8.set_ylabel('Input', fontsize=fontsize)

    x = np.arange(-.1, .1, .01)
    y1 = phi(x-.05, gain=10, p=1)
    y2 = phi(x-.015, gain=10, p=2)

    ax9.plot(x, y1, color=colors[1], linewidth=2)
    ax9.plot(x, y2, '--', color=colors[1], linewidth=2)

    ax9.set_xlabel('Potential (mV)')
    ax9.set_ylabel('Rate (kHz)')


    with open('/Users/gabeo/Dropbox (BOSTON UNIVERSITY)/Papers/weight_dynamics/Code/stdp_triplet_d=1_linear_neuron_data_orth_error.pkl', 'rb') as handle:
        datafile = pickle.load(handle)

    theta = datafile['angle_factor_weights']
    err_mean = np.mean(theta, axis=1)
    err_std = np.std(theta, axis=1) / np.sqrt(theta.shape[1])
    
    tstop = 10000
    tplot = np.arange(0, tstop, tstop//1000) / 1000 # 1000 points and convert ms to s

    ax10.fill_between(tplot, err_mean-err_std, err_mean+err_std, color=colors[0], alpha=0.2)
    ax10.plot(tplot, err_mean, linewidth=2, color=colors[0])

    with open('/Users/gabeo/Dropbox (BOSTON UNIVERSITY)/Papers/weight_dynamics/Code/stdp_pair_d=1_linear_neuron_data_orth_error.pkl', 'rb') as handle:
        datafile = pickle.load(handle)

    theta = datafile['angle_factor_weights']
    err_mean = np.mean(theta, axis=1)
    err_std = np.std(theta, axis=1) / np.sqrt(theta.shape[1])
    
    tstop = 200000
    tplot = np.arange(0, tstop, tstop//1000) / 1000 # 1000 points and convert ms to s

    ax10.fill_between(tplot, err_mean-err_std, err_mean+err_std, color=colors[1], alpha=0.2)
    ax10.plot(tplot, err_mean, linewidth=2, color=colors[1])


    with open('/Users/gabeo/Dropbox (BOSTON UNIVERSITY)/Papers/weight_dynamics/Code/stdp_pair_d=2_quadratic_neuron_data_orth_error.pkl', 'rb') as handle:
        datafile = pickle.load(handle)

    theta = datafile['angle_factor_weights']
    err_mean = np.mean(theta, axis=1)
    err_std = np.std(theta, axis=1) / np.sqrt(theta.shape[1])
    
    tstop = 200000
    tplot = np.arange(0, tstop, tstop//1000) / 1000 # 1000 points and convert ms to s

    ax10.fill_between(tplot, err_mean-err_std, err_mean+err_std, color=colors[1], alpha=0.2)
    ax10.plot(tplot, err_mean, '--', linewidth=2, color=colors[1])

    ax10.set_yticks((0, np.pi/4, np.pi/2))
    ax10.set_yticklabels([0, r'$\pi/4$', r'$\pi/2$'])
    ax10.set_ylabel('Error (rad.)', fontsize=fontsize)
    ax10.set_xscale('log')
    ax10.set_xlim((.01, 200))

    ax10.set_xlabel('Time (s)')

    sns.despine(fig)
    fig.tight_layout()

    fig.savefig(savefile)

if __name__ == '__main__':
    plot_fig_triplet_stdp(Npix=35, savefile='fig_tripletstdp.pdf')


    # print('pair stdp, linear neuron')
    # par = stdp_pair_params.params()
    # par.b = .05 * np.ones((par.N,))
    # par.p = 1
    # tstop = 200000
    # loop_initial_cond(par, Npix=35, tstop=tstop, stdp='pair', datafile_head='_linear_neuron_data_orth_error')

    # print('pair stdp, quadratic neuron')
    # par = stdp_pair_params.params()
    # par.b = .2 * np.ones((par.N,))
    # par.p = 2
    # tstop = 50000
    # loop_initial_cond(par, Npix=35, tstop=tstop, stdp='pair', datafile_head='_quadratic_neuron_data_orth_error')

    # print('triplet stdp, linear neuron')
    # par = stdp_triplet_params.params()
    # par.b = .015 * np.ones((par.N,))
    # par.p = 1
    # tstop = 10000
    # loop_initial_cond(par, Npix=35, tstop=tstop, stdp='triplet', datafile_head='_linear_neuron_data_orth_error')


    ### This creates a 5point correlation - ridiculous!
    # # print('triplet stdp, quadratic neuron')
    # # par = stdp_triplet_params.params()
    # # par.b = .1 * np.ones((par.N,))
    # # par.p = 2
    # # tstop = 50000
    # # loop_initial_cond(par, Npix=35, tstop=tstop, stdp='triplet', datafile_head='_quadratic_neuron_data_orth_error')
