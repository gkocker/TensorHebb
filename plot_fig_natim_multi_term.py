import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import central_diff_weights
from PIL import Image, ImageOps
from inequality.gini import Gini as gini
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import convolve2d
from sklearn import preprocessing
import tensortools as tt
import tensorly as tl
from tensorly.decomposition import tucker
import sys, os, pickle
import seaborn as sns

from plot_fig_natim import generate_inputs, unit_vector, angle_between, get_closest_factor, plot_image_stack, plot_orthog_approx_error, plot_3pt_corr, running_mean

fontsize = 10
labelsize = 8

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ['k']+colors


def generate_inputs_compute_factors(xx=10, yy=10, num_inputs=200, rank=20, a=[1, 2], b=[1, 1], savefile='inputs_factors_multiterm.pkl'):


    results_dict = {}

    inputs = generate_inputs(xx=xx, yy=yy, num_inputs=num_inputs)
    results_dict['inputs'] = inputs

    a = list(a)
    b = list(b)

    if len(a) > len(b):
        print('Mismatched parameter lengths, repeating last element of b to match length of a')
        b = b + [b[-1]]*(len(a)-len(b))
    
    elif len(b) > len(a):
        print('Mismatched parameter lengths, repeating last element of a to match length of b')
        a = a + [a[-1]]*(len(b)-len(a))

    else: pass

    N = len(a)
    for n in range(N):

        if a[n] == 3:
            mu = np.einsum('ij,ik,il,im',inputs**b[n],inputs,inputs,inputs) / num_inputs
        elif a[n] == 2:
            mu = np.einsum('ij,ik,il',inputs**b[n],inputs,inputs) / num_inputs
        elif a[n] == 1:
            mu = np.einsum('ij,ik',inputs**b[n],inputs) / num_inputs
        else:
            raise Exception('write input correlation computation for a[n]={}'.format(a[n]))

        if a[n] > 1:
            w, f = tucker(mu, rank=rank, verbose=True)
    
            # f = f[0]
            lam = np.linalg.norm(tl.unfold(w, 0), axis=1)
            maxind = np.argmax(np.abs(lam))

            if lam[maxind] < 0:
                lam *= -1
                v = f[0] * -1
            else: 
                v = f[0]

            ind = np.argsort(lam)[::-1]
            lam = lam[ind]
            v = v[:, ind]

            results_dict['singular_values_mode0_a={}_b={}'.format(a[n], b[n])] = lam
            results_dict['singular_vectors_mode0_a={}_b={}'.format(a[n], b[n])] = v

            if b[n] > 1:
                lam = np.linalg.norm(tl.unfold(w, 1), axis=1)
                maxind = np.argmax(np.abs(lam))

                if lam[maxind] < 0:
                    lam *= -1
                    v = f[1] * -1
                else: 
                    v = f[1]

                ind = np.argsort(lam)[::-1]
                lam = lam[ind]
                v = v[:, ind]

                results_dict['singular_values_mode1_a={}_b={}'.format(a[n], b[n])] = lam
                results_dict['singular_vectors_mode1_a={}_b={}'.format(a[n], b[n])] = v

        else:
            lam, v = np.linalg.eigh(mu)

            maxind = np.argmax(np.abs(lam))

            if lam[maxind] < 0:
                lam *= -1
                v *= -1

            ind = np.argsort(lam)[::-1]
            lam = lam[ind]
            v = v[:, ind]

            results_dict['singular_values_mode0_a={}_b={}'.format(a[n], b[n])] = lam
            results_dict['singular_vectors_mode0_a={}_b={}'.format(a[n], b[n])] = v


    with open(savefile, 'wb') as handle:
        pickle.dump(results_dict, handle)

    return inputs, lam, f



def run_sim(inputs, Nt=100000, eta=0.01, a=[1,2], b=[1,1], c=[0,0], A=[-1,1], p=2):

    if len(inputs.shape) != 2:
        raise Exception('Input array should be 2d, is {}d'.format(len(inputs.shape)))

    N = len(A)
    if (N != len(a)) or (N != len(b)) or (N != len(c)):
        raise Exception('Mismatched learning rule parameter lengths: {},{},{},{}'.format(len(A), len(a), len(b), len(c)))

    num_inputs, K = inputs

    J = np.random.randn(K)
    J /= np.linalg.norm(J, ord=p)

    for t in range(1, Nt):
                    
        # draw input pattern
        x = inputs[np.random.choice(num_inputs)]
        
        # update weights
        n = J.dot(x)
        F = np.zeros(K,)
        for m in range(N):
            F += A[m] * n**a[m] * x**b[m] * J**c[m]
        
        dJ = F - J * (F * J * np.abs(J)**(p-2)).sum()
        J = J + eta * dJ
        
    #     negind = np.where(Jt[t] < 0)[0]
    #     Jt[t, negind] = 0.
        
        if np.any(~np.isfinite(J)):
            break
            
    print('stop at t={}/{}'.format(t+1, Nt))

    return J


def run_sim_return_traces(inputs, Nt=1000000, eta=0.001, a=[1,2], b=[1,1], c=[0,0], A=[-1,1], p=2, J0=None, verbose=True):

    if len(inputs.shape) != 2:
        raise Exception('Input array should be 2d, is {}d'.format(len(inputs.shape)))

    num_inputs, K = inputs.shape
    
    N = len(A)
    if (N != len(a)) or (N != len(b)) or (N != len(c)):
        raise Exception('Mismatched learning rule parameter lengths: {},{},{},{}'.format(len(A), len(a), len(b), len(c)))

    if J0 is None:
        J0 = np.random.randn(K)
        J0 /= np.linalg.norm(J0, ord=p)

    Jt = np.zeros((Nt, K))
    Jt[0] = J0
    x_t = []

    for t in range(1, Nt):
            
        J = Jt[t-1]
        
        # draw input pattern
        x = inputs[np.random.choice(num_inputs)]
        x_t.append(x)
        
        # update weights
        n = J.dot(x)
        F = np.zeros(K,)
        for m in range(N):
            F += A[m] * (n**a[m]) * (x**b[m]) * (J**c[m])
        
        dJ = F - J * (F * J * np.abs(J)**(p-2)).sum()
        Jt[t] = J + eta * dJ
        
    #     negind = np.where(Jt[t] < 0)[0]
    #     Jt[t, negind] = 0.
        
        if np.any(~np.isfinite(Jt[t])):
            break
    
    if verbose:
        print('stop at t={}/{}'.format(t+1, Nt))

    return Jt, x_t


def loop_initial_cond(Nt=100000, Ninit=20, eta=0.01, a=[1,2], b=[1,1], c=[0,0], A=[-1,1], p=2, Npix=10, datafile_head='fig2_data_orth_error', rerun=False, eig_init=False):

    datafile = datafile_head+'_A={}_a={}_b={}_c={}.pkl'.format(A, a, b, c)
    if os.path.exists(datafile):
        print('Model ensemble already exists at {}'.format(datafile))
        
        if not rerun:
            with open(datafile, 'rb') as handle:
                results_dict = pickle.load(handle)

            return results_dict['angle_factor_weights']

        else: os.remove(datafile)

    else: pass

    theta = np.zeros((Nt, Ninit))
    norm = np.zeros((Nt, Ninit))
    Jdot = np.zeros((Nt-2, Ninit))
    factors = np.zeros((Npix**2, Ninit))

    ahat = max(a)
    N = len(A)
    if (N != len(a)) or (N != len(b)) or (N != len(c)):
        raise Exception('Mismatched learning rule parameter lengths: {},{},{},{}'.format(len(A), len(a), len(b), len(c)))

    results_dict = {}
    results_dict['a'] = a
    results_dict['b'] = b
    results_dict['c'] = c
    results_dict['p'] = p
    results_dict['A'] = A
    results_dict['Npix'] = Npix

    weights = central_diff_weights(3)

    for n in range(Ninit):

        inputs = generate_inputs(xx=Npix, yy=Npix)
        num_inputs, K = inputs.shape

        M = np.zeros([K for m in range(ahat+1)])
        
        for m in range(N):
            if a[m] == 1:
                mu = A[m] * np.einsum('ij,ik',inputs**b[m],inputs) / num_inputs  
                M[:, :, 0] += mu
            elif a[m] == 2:
                mu = A[m] * np.einsum('ij,ik,il',inputs**b[m],inputs,inputs) / num_inputs
                M[:, :, :] += mu
            else:
                print('write input correlation computation for m={}'.format(m))

        U = tt.cp_als(M, rank=1, verbose=False)

        if eig_init:
            factor = np.squeeze(U.factors[0])
            J0 = factor + .1*np.random.randn(K,)
            J0 /= np.linalg.norm(J0, ord=p)
            Jt, _ = run_sim_return_traces(inputs, Nt=Nt, eta=eta, a=a, b=b, c=c, p=p, J0=J0)
        
        else:
            Jt, _ = run_sim_return_traces(inputs, Nt=Nt, eta=eta, a=a, b=b, c=c, p=p)
            Jplot = np.mean(Jt[-1000:, :], axis=0)
            factor = get_closest_factor(U, Jplot, ahat)

        for t in range(Nt):
            theta[t, n] = angle_between(Jt[t], factor)

        Jdot_tmp = []
        for t in range(1, Nt-1):
            vals = np.concatenate((Jt[t-1][:, None], Jt[t][:, None], Jt[t+1][:, None]), axis=1)
            Jdot_tmp.append(vals.dot(weights))
        
        Jdot_tmp = np.array(Jdot_tmp)
        Jdot[:, n] = np.amax(np.abs(Jdot_tmp), axis=1)

        norm[:, n] = np.linalg.norm(Jt, axis=1, ord=p)
        factors[:, n] = factor

    results_dict['factors'] = factors
    results_dict['angle_factor_weights'] = theta
    results_dict['Jdot'] = Jdot
    results_dict['norm'] = norm

    with open(datafile, 'wb') as handle:
        pickle.dump(results_dict, handle)

    return theta


def run_orthog_approx_error(a=[1, 2], b=[1, 1], A=[-1, 1], max_rank=4, Npix=10, replicates=4, datafile_head='fig2_data_ensemble', rerun=False):

    datafile = datafile_head+'_a={}_b={}_A={}.pkl'.format(a, b, A)
    if os.path.exists(datafile):
        print('Model ensemble already exists at {}'.format(datafile))
        
        if not rerun:
            with open(datafile, 'rb') as handle:
                results_dict = pickle.load(handle)

            return results_dict['objectives']

        else: pass

    else: pass

    N = len(A)
    if (N != len(a)) or (N != len(b)):
        raise Exception('Mismatched learning rule parameter lengths: {},{},{}'.format(len(A), len(a), len(b)))

    inputs = generate_inputs(xx=Npix, yy=Npix)
    num_inputs, K = inputs.shape
    ranks = range(1, max_rank)

    ahat = max(a)
    M = np.zeros([K for m in range(ahat+1)])
    
    for m in range(N):
        if a[m] == 1:
            mu = A[m] * np.einsum('ij,ik',inputs**b[m],inputs) / num_inputs  
            M[:, :, 0] += mu
        elif a[m] == 2:
            mu = A[m] * np.einsum('ij,ik,il',inputs**b[m],inputs,inputs) / num_inputs
            M[:, :, :] += mu
        else:
            print('write input correlation computation for m={}, a[m]={}'.format(m, a[m]))

    del mu

    ensemble = tt.Ensemble(fit_method="cp_als")
    ensemble.fit(M, ranks=ranks, replicates=replicates)

    objectives = []
    similarities = []
    for r in ranks:
        objectives.append(ensemble.objectives(rank=r))
        similarities.append(ensemble.similarities(rank=r))

    results_dict = {}
    # results_dict['ensemble'] = ensemble
    results_dict['a'] = a
    results_dict['b'] = b
    results_dict['Npix'] = Npix
    results_dict['ranks'] = ranks
    results_dict['objectives'] = objectives
    results_dict['similarities'] = similarities

    with open(datafile, 'wb') as handle:
        pickle.dump(results_dict, handle)

    del M
    del ensemble

    return objectives


def run_asymmetry_error(a=[1, 2], b=[1, 1], A1=-1, A2=np.arange(-3, 3.2, .2), Ninit=20, Npix=10, datafile_head='fig2_data_corr_asymmetry', rerun=False):

    datafile = datafile_head+'_a={}_b={}_A1={}.pkl'.format(a, b, A1)
    if os.path.exists(datafile):
        print('Model ensemble already exists at {}'.format(datafile))
        
        if not rerun:
            with open(datafile, 'rb') as handle:
                results_dict = pickle.load(handle)

            return results_dict['objectives']

        else: pass

    else: pass

    # N = len(A)
    # if (N != len(a)) or (N != len(b)):
    #     raise Exception('Mismatched learning rule parameter lengths: {},{},{}'.format(len(A), len(a), len(b)))

    NA = len(A2)
    err = np.zeros(NA, Ninit)

    for n in range(Ninit):

        inputs = generate_inputs(xx=Npix, yy=Npix)
        num_inputs, K = inputs.shape
        ahat = max(a)

        for i, A2i in enumerate(A2):
            A = [A1, A2i]

            M = np.zeros([K for m in range(ahat+1)])
            
            for m in range(N):
                if a[m] == 1:
                    mu = A[m] * np.einsum('ij,ik',inputs**b[m],inputs) / num_inputs  
                    M[:, :, 0] += mu
                elif a[m] == 2:
                    mu = A[m] * np.einsum('ij,ik,il',inputs**b[m],inputs,inputs) / num_inputs
                    M[:, :, :] += mu
                else:
                    print('write input correlation computation for m={}, a[m]={}'.format(m, a[m]))

            del mu
            del M 

            err[i, n] = np.mean((M - M.T)**2)

    results_dict = {}
    # results_dict['ensemble'] = ensemble
    results_dict['a'] = a
    results_dict['b'] = b
    results_dict['A1'] = A1
    results_dict['A2'] = A2
    results_dict['Npix'] = Npix
    results_dict['symmetry_error'] = err

    with open(datafile, 'wb') as handle:
        pickle.dump(results_dict, handle)

    return results_dict


def plot_combined_3pt_corr(ax, inputs, a=[1, 2], b=[1, 1], A=[-1, 1], Nplot=4, Nsmooth=31, cmap='cividis', cscale=.35):

    N = len(A)
    if (N != len(a)) or (N != len(b)):
        raise Exception('Mismatched learning rule parameter lengths: {},{},{},{}'.format(len(A), len(a), len(b), len(c)))

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
    plot_ind = list(range(0, K, K//(Nplot)))
    M = np.zeros([Kplot for m in range(ahat)]+[Nplot])
    
    for m in range(N):
        inputsb = inputs**b[m]

        if a[m] == 1:
            mui = A[m] * np.einsum('ij,ik',inputs**b[m],inputs) / num_inputs
            mui = convolve2d(mui, np.ones((Nsmooth, Nsmooth)), mode='same', boundary='wrap')
            mui = mui[::Nsmooth//2, ::Nsmooth//2]

            M[:, :, 0] += mui

        elif a[m] == 2:
            for i, z in enumerate(plot_ind):
                inputs2 = inputs * np.outer(inputs[:, z], np.ones(K,))
                mui = A[m] * inputsb.T.dot(inputs2) / num_inputs
                mui = convolve2d(mui, np.ones((Nsmooth, Nsmooth)), mode='same', boundary='wrap')
                mui = mui[::Nsmooth//2, ::Nsmooth//2]

                M[:, :, i] += mui

        else:
            print('write input correlation computation for m={}, a[m]={}'.format(m, a[m]))

    del mui

    cmid = np.mean(M)
    cmin = cmid - (np.amax(M) - np.amin(M))*cscale
    cmax = cmid + (np.amax(M) - np.amin(M))*cscale

    scam = plt.cm.ScalarMappable(
        norm=plt.cm.colors.Normalize(cmin, cmax),
        cmap=cmap
    )

    for z in range(Nplot):
        Zmesh = z * np.ones((Kplot, Kplot))
        
        scam.set_array([])   
        surf = ax.plot_surface(
            Xmesh, Ymesh, Zmesh,
            facecolors  = scam.to_rgba(M[:, :, z]),
            antialiased = True,
            rstride=1, cstride=1, alpha=None
        )

    return scam, plot_ind


def plot_fig_2(savefile='fig2.pdf', datafile='fig2_data.pkl', cmap='cividis', max_rank=4, Npix=10, Ninit=20, eta=.01, Nplot=10, data_dir="/Users/gabeo/Dropbox (BOSTON UNIVERSITY)/Papers/tensor_hebb/Code"):

    fig = plt.figure(figsize=(5.67, 3.7))
    
    if (sys.platform != 'darwin') and ('win' in sys.platform):
        im_dir = "C:/Users/Gabe/Documents/projects/images/berkeley_segmentation/train"
        inputs = generate_inputs(xx=Npix, yy=Npix, im_dir=im_dir)
    else:
        inputs = generate_inputs(xx=Npix, yy=Npix)

    num_inputs, K = inputs.shape

    a = [1, 2]
    b = [1, 1]
    c = [0, 0]
    A = [-2, 1]
    N = len(a)
    ahat = max(a)

    ### example pair and triplet correlations
    ax1 = fig.add_subplot(2, 3, 1)
    # mu = np.einsum('ij,ik', inputs, inputs) / num_inputs
    # ax1.imshow(mu, cmap=cmap)

    ax2 = fig.add_subplot(2, 3, 4)
    # scam, plot_ind = plot_3pt_corr(ax2, inputs=inputs, b=b, cscale=0.25, cmap=cmap)

    ### asymmetry of the correlations that plasticity decomposes
    ax3 = fig.add_subplot(2, 3, 2)

    # datafile = os.path.join(data_dir, "fig2_data_corr_asymmetry_a=[1, 2]_b=[1, 1]_A1=-1.pkl")
    # with open(datafile, 'rb') as handle:
    #     results_dict = pickle.load(handle)

    # A2 = results_dict['A2']
    # symm_err = results_dict['symmetry_error']

    # error_mean = symm_err.mean(axis=1)
    # error_std = symm_err.std(axis=1) / symm_err.shape[1]
    # ax3.fill_between(error_mean + error_std, error_mean- error_std, alpha=0.2, color=colors[0])
    # ax3.plot(A2, error_mean, linewidth=2, color=colors[0])

    ### example dynamics
    ax4 = fig.add_subplot(2, 3, 5)

    Npix = 35
    if (sys.platform != 'darwin') and ('win' in sys.platform):
        im_dir = "C:/Users/Gabe/Dropbox (BOSTON UNIVERSITY)/images/berkeley_segmentation/train"
        inputs = generate_inputs(xx=Npix, yy=Npix, im_dir=im_dir)
    else:
        inputs = generate_inputs(xx=Npix, yy=Npix)

    Jt, _ = run_sim_return_traces(inputs)



    ### convergence: Jdot, norm and alignment with rank one approximation

    ax6 = fig.add_subplot(3, 3, 3)
    ax7 = fig.add_subplot(3, 3, 6)
    ax8 = fig.add_subplot(3, 3, 9)
    



    # ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    # scam, plot_ind = plot_combined_3pt_corr(ax1, inputs=inputs, a=a, b=b, A=A, cscale=0.25, cmap=cmap)    
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    # ax1.set_zticks([])
    # ax1.set_xlabel('Pixel', fontsize=fontsize)
    # ax1.set_ylabel('Pixel', fontsize=fontsize)
    # ax1.set_zlabel('Pixel', fontsize=fontsize)
    # ax1.set_title('3-point corr.\n(A,a)={},{}'.format(A,a), fontsize=fontsize)

    # ax2 = fig.add_subplot(2, 2, 2)
    # marker_style = {'s':20, 'c':colors[0]}
    # line_style = {'linewidth':3,'color':colors[0]}

    # datafile = os.path.join(data_dir, "fig2_data_ensemble_a=[1, 2]_b=[1, 1]_A=[-2, 1].pkl")
    # with open(datafile, 'rb') as handle:
    #     results_dict = pickle.load(handle)
    # _ = plot_orthog_approx_error(ax2, results_dict, color=colors[0], label='A=[-2,1]')

    # datafile = os.path.join(data_dir, "fig2_data_ensemble_a=[1, 2]_b=[1, 1]_A=[-1, 1].pkl")
    # with open(datafile, 'rb') as handle:
    #     results_dict = pickle.load(handle)
    # _ = plot_orthog_approx_error(ax2, results_dict, color=colors[1], label='A=[-1,1]'.format(A))

    # ax2.set_xlabel('Rank', fontsize=fontsize)
    # ax2.set_ylabel('MSE of orth.\napproximation', fontsize=fontsize)
    # ax2.legend(loc=0, frameon=False, fontsize=fontsize)

    # ax2.set_ylim((0, ax2.get_ylim()[1]))

    # num_inputs, K = inputs.shape

    # Nt = 5000
    # tplot = range(0, Nt, Nt//100)
    
    # N = len(a)
    # ahat = max(a)
    # Jt, _ = run_sim_return_traces(inputs=inputs, a=a, b=b, c=c, A=A, Nt=Nt, eta=eta)

    # # Jplot = np.mean(Jt[-5000:, :], axis=0)
    # # factor = get_closest_factor(U, Jplot, max(a))

    # plot_ind = np.random.choice(K, Nplot)

    # ax3 = fig.add_subplot(2, 2, 3)
    # ax3.plot(tplot, Jt[tplot][:, plot_ind], linewidth=2)

    # ax3.set_xlabel('Time', fontsize=fontsize)
    # ax3.set_ylabel('Synaptic weight', fontsize=fontsize)
    
    # ax4 = fig.add_subplot(2, 2, 4)
    # datafile = os.path.join(data_dir, "fig2_data_orth_error_A=[-2, 1]_a=[1, 2]_b=[1, 1]_c=[0, 0].pkl")
    # with open(datafile, 'rb') as handle:
    #     results_dict = pickle.load(handle)

    # T = results_dict['angle_factor_weights'].shape[0]
    # Ntplot = 200
    # tplot = range(0, T, T//Ntplot)

    # error_angle = results_dict['angle_factor_weights']
    # error_std = np.nanstd(error_angle, axis=1) / np.sqrt(Ninit)
    # error_mean = np.nanmean(error_angle, axis=1)

    # ax4.fill_between(tplot, error_mean[tplot] + error_std[tplot], error_mean[tplot] - error_std[tplot], alpha=0.2, color=colors[0])
    # ax4.plot(tplot, error_mean[tplot], linewidth=2, color=colors[0])
    

    # datafile = os.path.join(data_dir, "fig2_data_orth_error_A=[-2, 1]_a=[1, 2]_b=[1, 1]_c=[0, 0].pkl")
    # with open(datafile, 'rb') as handle:
    #     results_dict = pickle.load(handle)

    # T = results_dict['angle_factor_weights'].shape[0]
    # Ntplot = 200
    # tplot = range(0, T, T//Ntplot)

    # error_angle = results_dict['angle_factor_weights']
    # error_std = np.nanstd(error_angle, axis=1) / np.sqrt(Ninit)
    # error_mean = np.nanmean(error_angle, axis=1)

    # ax4.fill_between(tplot, error_mean[tplot] + error_std[tplot], error_mean[tplot] - error_std[tplot], alpha=0.2, color=colors[0])
    # ax4.plot(tplot, error_mean[tplot], linewidth=2, color=colors[0])
    

    # datafile = os.path.join(data_dir, "fig2_data_orth_error_A=[-1, 1]_a=[1, 2]_b=[1, 1]_c=[0, 0].pkl")
    # with open(datafile, 'rb') as handle:
    #     results_dict = pickle.load(handle)

    # T = results_dict['angle_factor_weights'].shape[0]
    # Ntplot = 200
    # tplot = range(0, T, T//Ntplot)

    # error_angle = results_dict['angle_factor_weights']
    # error_std = np.nanstd(error_angle, axis=1) / np.sqrt(Ninit)
    # error_mean = np.nanmean(error_angle, axis=1)

    # ax4.fill_between(tplot, error_mean[tplot] + error_std[tplot], error_mean[tplot] - error_std[tplot], alpha=0.2, color=colors[1])
    # ax4.plot(tplot, error_mean[tplot], linewidth=2, color=colors[1])
    

    # datafile = os.path.join(data_dir, "fig2_data_orth_error_eig_init_A=[-1, 1]_a=[1, 2]_b=[1, 1]_c=[0, 0].pkl")
    # with open(datafile, 'rb') as handle:
    #     results_dict = pickle.load(handle)

    # T = results_dict['angle_factor_weights'].shape[0]
    # Ntplot = 200
    # tplot = range(0, T, T//Ntplot)

    # error_angle = results_dict['angle_factor_weights']
    # error_std = np.nanstd(error_angle, axis=1) / np.sqrt(Ninit)
    # error_mean = np.nanmean(error_angle, axis=1)

    # ax4.fill_between(tplot, error_mean[tplot] + error_std[tplot], error_mean[tplot] - error_std[tplot], alpha=0.2, color=colors[2])
    # ax4.plot(tplot, error_mean[tplot], linewidth=2, color=colors[2])


    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(savefile)

    return None


def plot_fig_natural_images_multi_term(savefile='fig_natim_mult_term.pdf', cmap='cividis', max_rank=4, Npix=10, Ninit=20, Nplot=10, data_dir="/Users/gabeo/Dropbox (BOSTON UNIVERSITY)/Papers/tensor_hebb/Code"):

    fig = plt.figure(figsize=(7, 3.7))

    a = [1, 2]
    b = [1, 1]
    c = [0, 0]
    p = 2
    N = len(a)

    ax1a = fig.add_subplot(2, 4, 1)
    ax2a = fig.add_subplot(2, 4, 5)

    input_files = [f for f in os.listdir(data_dir) if (f.split('_')[0] == 'inputs')]
    input_files = [f for f in input_files if ('multiterm' in f)]
    input_files = [f for f in input_files if not ('sim' in f)]
    num_input_samples = len(input_files)

    with open(os.path.join(data_dir, input_files[0]), 'rb') as handle:
        input_file = pickle.load(handle)

    factors1_tmp = input_file['singular_vectors_mode0_a={}_b={}'.format(a[0], b[0])] #* A1[0]
    lam1_tmp = input_file['singular_values_mode0_a={}_b={}'.format(a[0], b[0])]

    factors2_tmp = input_file['singular_vectors_mode0_a={}_b={}'.format(a[1], b[1])] #* A1[1]
    lam2_tmp = input_file['singular_values_mode0_a={}_b={}'.format(a[1], b[1])]

    Npix, Nfactors1 = factors1_tmp.shape
    Nfactors2 = factors2_tmp.shape[1]
    Nfactors = min(Nfactors1, Nfactors2)

    lam1 = np.zeros((Nfactors, num_input_samples))
    factors1 = np.zeros((Npix, Nfactors, num_input_samples))

    lam2 = np.zeros((Nfactors, num_input_samples))
    factors2 = np.zeros((Npix, Nfactors, num_input_samples))

    overlap = np.zeros((Nfactors, Nfactors, num_input_samples))

    lam1[:, 0] = lam1_tmp[:Nfactors]
    factors1[:, :, 0] = factors1_tmp[:, :Nfactors]

    lam2[:, 0] = lam2_tmp[:Nfactors]
    factors2[:, :, 0] = factors2_tmp[:, :Nfactors]

    overlap[:, :, 0] = np.abs(factors1[:, :, 0].T.dot(factors2[:, :, 0]))

    for i in range(1, num_input_samples):
        with open(os.path.join(data_dir, input_files[i]), 'rb') as handle:
            input_file = pickle.load(handle)
    
        lam1[:, i] = input_file['singular_values_mode0_a={}_b={}'.format(a[0], b[0])][:Nfactors]
        factors1[:, :, i] = input_file['singular_vectors_mode0_a={}_b={}'.format(a[0], b[0])][:, :Nfactors]

        lam2[:, i] = input_file['singular_values_mode0_a={}_b={}'.format(a[1], b[1])][:Nfactors]
        factors2[:, :, i] = input_file['singular_vectors_mode0_a={}_b={}'.format(a[1], b[1])][:, :Nfactors]

        overlap[:, :, i] = np.abs(factors1[:, :, i].T.dot(factors2[:, :, i]))

    lam1_mean = np.mean(lam1, axis=-1)
    lam1_std = np.std(lam1, axis=-1)

    lam2_mean = np.mean(lam2, axis=-1)
    lam2_std = np.std(lam2, axis=-1)

    ax1a.fill_between(range(Nfactors), lam1_mean-lam1_std, lam1_mean+lam1_std, color=colors[0], alpha=0.2)
    ax1a.plot(range(Nfactors), lam1_mean, color=colors[0], linewidth=2, label='2-point')

    ax1a.fill_between(range(Nfactors), lam2_mean-lam2_std, lam2_mean+lam2_std, color=colors[1], alpha=0.2)
    ax1a.plot(range(Nfactors), lam2_mean, color=colors[1], linewidth=2, label='2-point')

    overlap_plot = ax2a.imshow(overlap.mean(axis=-1)[:10, :10], cmap='cividis')
    fig.colorbar(overlap_plot, ax=ax2a)

    A1 = [1, .5] # strong 2pt contribution, weak 3pt contribution
    A2 = [.5, 1] # weak 2pt contribution, strong 3pt contribution

    ax1 = fig.add_subplot(3, 4, 2)
    ax2 = fig.add_subplot(3, 4, 6)
    ax3 = fig.add_subplot(3, 4, 10)

    input_files = [f for f in os.listdir(data_dir) if (f.split('_')[0] == 'inputs')]
    input_files = [f for f in input_files if ('multiterm' in f)]
    input_files = [f for f in input_files if not ('sim' in f)]
    num_input_samples = len(input_files)
    print(input_files)

    with open(os.path.join(data_dir, input_files[0]), 'rb') as handle:
        input_file = pickle.load(handle)

    inputs = input_file['inputs']
    num_inputs, K = inputs.shape

    dt_ds = 100
    Nt = 15000
    tplot = np.arange(0, Nt, dt_ds)

    ### example with strong 2-point contribution
    Jt, _ = run_sim_return_traces(inputs, A=A1, a=a, b=b, c=c, p=p, Nt=Nt)
    # Jt = Jt[::dt_ds]
    
    plot_ind = np.random.choice(K, 8)
    for i in plot_ind:
        Jt_i = Jt[:, i]
        Jt_i_plot = running_mean(Jt_i, N=dt_ds)
        ax1.plot(tplot, Jt_i_plot[::dt_ds], linewidth=1, color='k')

    ### projection onto eigenvectors of first input correlation
    factors = input_file['singular_vectors_mode0_a={}_b={}'.format(a[0], b[0])] * A1[0]
    lam = input_file['singular_values_mode0_a={}_b={}'.format(a[0], b[0])]

    norms = np.linalg.norm(factors, axis=0) # norm across components of each factor
    for i in range(factors.shape[1]): # range over factors
        factors[:, i] /= norms[i]

    lam *= norms

    thetai = np.abs(np.dot(Jt, factors))
    fac_ind = np.argmax(np.mean(thetai[-10:], axis=0))

    for i in range(5):
        thetai_plot = running_mean(thetai[:, i], N=dt_ds)
        ax2.plot(tplot, thetai_plot[::dt_ds], linewidth=1, color='k', alpha=1/(i+1))

    ### projection onto eigenvectors of second input correlation
    factors = input_file['singular_vectors_mode0_a={}_b={}'.format(a[1], b[1])] * A1[1]
    lam = input_file['singular_values_mode0_a={}_b={}'.format(a[1], b[1])]

    norms = np.linalg.norm(factors, axis=0) # norm across components of each factor
    for i in range(factors.shape[1]): # range over factors
        factors[:, i] /= norms[i]

    lam *= norms

    thetai = np.abs(np.dot(Jt, factors))
    fac_ind = np.argmax(np.mean(thetai[-10:], axis=0))

    for i in range(5):
        thetai_plot = running_mean(thetai[:, i], N=dt_ds)
        ax3.plot(tplot, thetai_plot[::dt_ds], linewidth=1, color=colors[1], alpha=1/(i+1))


    ### histogram of final factors
    Nt = 50000
    ax7 = fig.add_subplot(2, 4, 3)
    ax8 = fig.add_subplot(2, 4, 7, sharex=ax7, sharey=ax7)

    num_its_per_sample = 10

    sim_file = 'inputs_factors_Npix35_a2_sims_multiterm_A=({},{}).pkl'.format(A1[0],A1[1])

    if os.path.exists(os.path.join(data_dir, sim_file)):
        with open(os.path.join(data_dir, sim_file), 'rb') as handle:
            overlap_dict = pickle.load(handle)
    else:
        print('Running sims for factor overlap')

        theta_1 = np.zeros((Nt//dt_ds, num_input_samples, num_its_per_sample))
        factor_num_1 = np.zeros((num_input_samples, num_its_per_sample))

        theta_2 = np.zeros((Nt//dt_ds, num_input_samples, num_its_per_sample))
        factor_num_2 = np.zeros((num_input_samples, num_its_per_sample))

        for i, f in enumerate(input_files):

            with open(os.path.join(data_dir, f), 'rb') as handle:
                input_file = pickle.load(handle)

            inputs = input_file['inputs']

            factors1 = input_file['singular_vectors_mode0_a={}_b={}'.format(a[0], b[0])] * A1[0]
            lam1 = input_file['singular_values_mode0_a={}_b={}'.format(a[0], b[0])]

            norms = np.linalg.norm(factors1, axis=0) # norm across components of each factor
            for k in range(factors1.shape[1]): # range over factors
                factors1[:, k] /= norms[k]
            lam1 *= norms

            factors2 = input_file['singular_vectors_mode0_a={}_b={}'.format(a[1], b[1])] * A1[1]
            lam2 = input_file['singular_values_mode0_a={}_b={}'.format(a[1], b[1])]

            norms = np.linalg.norm(factors2, axis=0) # norm across components of each factor
            for k in range(factors2.shape[1]): # range over factors
                factors2[:, k] /= norms[k]
            lam2 *= norms

            for j in range(num_its_per_sample):
                Jt, _ = run_sim_return_traces(inputs, A=A1, a=a, b=b, c=c, p=p, Nt=Nt)
                # Jt = Jt[::dt_ds]
                
                thetai = np.abs(np.dot(Jt, factors1))
                fac_ind = np.argmax(np.mean(thetai[-5000:], axis=0))
                
                thetai_plot = running_mean(thetai[:, 0], N=dt_ds)
                theta_1[:, i, j] = thetai_plot[::dt_ds]
                factor_num_1[i, j] = fac_ind

                thetai = np.abs(np.dot(Jt, factors2))
                fac_ind = np.argmax(np.mean(thetai[-5000:], axis=0))
                
                thetai_plot = running_mean(thetai[:, 0], N=dt_ds)
                theta_2[:, i, j] = thetai_plot[::dt_ds]

                factor_num_2[i, j] = fac_ind

        overlap_dict = {}
        overlap_dict['overlap_1'] = theta_1
        overlap_dict['overlap_2'] = theta_2
        overlap_dict['factor_number_1'] = factor_num_1
        overlap_dict['factor_number_2'] = factor_num_2

        with open(os.path.join(data_dir, sim_file), 'wb') as handle:
            pickle.dump(overlap_dict, handle)

    # factor_num_1 = overlap_dict['factor_number_1']
    # factor_num_2 = overlap_dict['factor_number_2']
    # bins = range(0, K)

    # ax7.hist(factor_num_1.reshape(-1,), bins=bins, align='mid', density=True, alpha=0.5, label=r'$a_1={}$'.format(a[0]))
    # ax7.hist(factor_num_2.reshape(-1,), bins=bins, align='mid', density=True, alpha=0.5, label=r'$a_2={}$'.format(a[1]))

    theta_1 = overlap_dict['overlap_1']
    theta_2 = overlap_dict['overlap_2']

    ax7.hist(theta_1[-1].reshape(-1,), bins = np.arange(0, 1.1, .1), align='mid', density=True, alpha=0.5, label=r'$A=({},{})$'.format(A1[0],A1[1]))
    ax8.hist(theta_2[-1].reshape(-1,),  bins = np.arange(0, 1.1, .1), align='mid', density=True, alpha=0.5, label=r'$A=({},{})$'.format(A1[0],A1[1]))

    ### example with weak 2-point contribution
    # ax4 = fig.add_subplot(3, 4, 3)
    # ax5 = fig.add_subplot(3, 4, 7)
    # ax6 = fig.add_subplot(3, 4, 11)

    # Nt = 15000
    # Jt, _ = run_sim_return_traces(inputs, A=A2, a=a, b=b, c=c, p=p, Nt=Nt)
    # # Jt = Jt[::dt_ds]
    
    # plot_ind = np.random.choice(K, 5)
    # for i in plot_ind:
    #     Jt_i = Jt[:, i]
    #     Jt_i_plot = running_mean(Jt_i, N=dt_ds)
    #     ax4.plot(tplot, Jt_i_plot[::dt_ds], linewidth=1, color='k')

    ### projection onto eigenvectors of first input correlation
    # factors = input_file['singular_vectors_mode0_a={}_b={}'.format(a[0], b[0])] * A2[0]
    # lam = input_file['singular_values_mode0_a={}_b={}'.format(a[0], b[0])]

    # norms = np.linalg.norm(factors, axis=0) # norm across components of each factor
    # for i in range(factors.shape[1]): # range over factors
    #     factors[:, i] /= norms[i]

    # lam *= norms

    # thetai = np.abs(np.dot(Jt, factors))
    # fac_ind = np.argmax(np.mean(thetai[-10:], axis=0))

    # for i in range(5):
    #     thetai_plot = running_mean(thetai[:, i], N=dt_ds)
    #     ax5.plot(tplot, thetai_plot[::dt_ds], linewidth=1, color='k', alpha=1/(i+1))

    # ### projection onto eigenvectors of second input correlation
    # factors = input_file['singular_vectors_mode0_a={}_b={}'.format(a[1], b[1])] * A2[1]
    # lam = input_file['singular_values_mode0_a={}_b={}'.format(a[1], b[1])]

    # norms = np.linalg.norm(factors, axis=0) # norm across components of each factor
    # for i in range(factors.shape[1]): # range over factors
    #     factors[:, i] /= norms[i]

    # lam *= norms

    # thetai = np.abs(np.dot(Jt, factors))
    # fac_ind = np.argmax(np.mean(thetai[-10:], axis=0))

    # for i in range(5):
    #     thetai_plot = running_mean(thetai[:, i], N=dt_ds)
    #     ax6.plot(tplot, thetai_plot[::dt_ds], linewidth=1, color='k', alpha=1/(i+1))
    

   ### histogram of final factors 
    Nt = 50000

    sim_file = 'inputs_factors_Npix35_a2_sims_multiterm_A=({},{}).pkl'.format(A2[0],A2[1])

    if os.path.exists(os.path.join(data_dir, sim_file)):
        with open(os.path.join(data_dir, sim_file), 'rb') as handle:
            overlap_dict = pickle.load(handle)
    else:
        print('Running sims for factor overlap')

        theta_1 = np.zeros((Nt//dt_ds, num_input_samples, num_its_per_sample))
        factor_num_1 = np.zeros((num_input_samples, num_its_per_sample))

        theta_2 = np.zeros((Nt//dt_ds, num_input_samples, num_its_per_sample))
        factor_num_2 = np.zeros((num_input_samples, num_its_per_sample))

        for i, f in enumerate(input_files):

            with open(os.path.join(data_dir, f), 'rb') as handle:
                input_file = pickle.load(handle)

            inputs = input_file['inputs']

            factors1 = input_file['singular_vectors_mode0_a={}_b={}'.format(a[0], b[0])] * A2[0]
            lam1 = input_file['singular_values_mode0_a={}_b={}'.format(a[0], b[0])]

            norms = np.linalg.norm(factors1, axis=0) # norm across components of each factor
            for k in range(factors1.shape[1]): # range over factors
                factors1[:, k] /= norms[k]
            lam1 *= norms

            factors2 = input_file['singular_vectors_mode0_a={}_b={}'.format(a[1], b[1])] * A2[1]
            lam2 = input_file['singular_values_mode0_a={}_b={}'.format(a[1], b[1])]

            norms = np.linalg.norm(factors2, axis=0) # norm across components of each factor
            for k in range(factors2.shape[1]): # range over factors
                factors2[:, k] /= norms[k]
            lam2 *= norms

            for j in range(num_its_per_sample):
                Jt, _ = run_sim_return_traces(inputs, A=A2, a=a, b=b, c=c, p=p, Nt=Nt)
                # Jt = Jt[::dt_ds]
                
                thetai = np.abs(np.dot(Jt, factors1))
                fac_ind = np.argmax(np.mean(thetai[-5000:], axis=0))

                thetai_plot = running_mean(thetai[:, 0], N=dt_ds)
                theta_1[:, i, j] = thetai_plot[::dt_ds]

                factor_num_1[i, j] = fac_ind

                thetai = np.abs(np.dot(Jt, factors2))
                fac_ind = np.argmax(np.mean(thetai[-5000:], axis=0))
                
                thetai_plot = running_mean(thetai[:, 0], N=dt_ds)
                theta_2[:, i, j] = thetai_plot[::dt_ds]

                factor_num_2[i, j] = fac_ind

        overlap_dict = {}
        overlap_dict['overlap_1'] = theta_1
        overlap_dict['overlap_2'] = theta_2
        overlap_dict['factor_number_1'] = factor_num_1
        overlap_dict['factor_number_2'] = factor_num_2

        with open(os.path.join(data_dir, sim_file), 'wb') as handle:
            pickle.dump(overlap_dict, handle)

    theta_1 = overlap_dict['overlap_1']
    theta_2 = overlap_dict['overlap_2']

    ax7.hist(theta_1[-1].reshape(-1,), bins = np.arange(0, 1.1, .1), align='mid', density=True, alpha=0.5, label=r'$A=({},{})$'.format(A2[0],A2[1]))
    ax8.hist(theta_2[-1].reshape(-1,), bins = np.arange(0, 1.1, .1), align='mid', density=True, alpha=0.5, label=r'$A=({},{})$'.format(A2[0],A2[1]))

 
    # # factor_num_1 = overlap_dict['factor_number_1']
    # # factor_num_2 = overlap_dict['factor_number_2']
    # # bins = range(0, K)
    # # ax8.hist(factor_num_1.reshape(-1,), bins=bins, align='mid', density=True, alpha=0.5, label=r'$a_1={}$'.format(a[0]))
    # # ax8.hist(factor_num_2.reshape(-1,), bins=bins, align='mid', density=True, alpha=0.5, label=r'$a_2={}$'.format(a[1]))



    ### range over 2-point weight
    ### plot final alignment with each factor

    eta = .001
    sim_file = 'inputs_factors_Npix35_a2_sims_multiterm_two_point_weight_range.pkl'.format(eta)

    if os.path.exists(os.path.join(data_dir, sim_file)):
        with open(os.path.join(data_dir, sim_file), 'rb') as handle:
            overlap_dict = pickle.load(handle)
    else:
        print('Running sims, ranging over two-point weight')

        two_point_weight_range = np.arange(-.5, 1.6, .1)
        Nweights = len(two_point_weight_range)
        
        Nt = 100000
        num_its_per_sample = 10
        num_factors = 20

        theta1 = np.zeros((Nweights, num_factors, num_input_samples, num_its_per_sample))
        theta2 = np.zeros((Nweights, num_factors, num_input_samples, num_its_per_sample))

        for m, Am in enumerate(two_point_weight_range):
            A = [Am, 1]
            print('{}/{}'.format(m, Nweights))

            for i, f in enumerate(input_files):

                with open(os.path.join(data_dir, f), 'rb') as handle:
                    input_file = pickle.load(handle)

                inputs = input_file['inputs']

                factors1 = input_file['singular_vectors_mode0_a={}_b={}'.format(a[0], b[0])] * A[0]
                lam1 = input_file['singular_values_mode0_a={}_b={}'.format(a[0], b[0])]

                norms = np.linalg.norm(factors1, axis=0) # norm across components of each factor
                for k in range(factors1.shape[1]): # range over factors
                    factors1[:, k] /= norms[k]
                lam1 *= norms

                factors2 = input_file['singular_vectors_mode0_a={}_b={}'.format(a[1], b[1])] * A[1]
                lam2 = input_file['singular_values_mode0_a={}_b={}'.format(a[1], b[1])]

                norms = np.linalg.norm(factors2, axis=0) # norm across components of each factor
                for k in range(factors2.shape[1]): # range over factors
                    factors2[:, k] /= norms[k]
                lam2 *= norms

                for j in range(num_its_per_sample):
                    Jt, _ = run_sim_return_traces(inputs, A=A, a=a, b=b, c=c, p=p, Nt=Nt, eta=eta, verbose=False)

                    thetai = np.abs(np.dot(Jt, factors1))
                    theta1[m, :, i, j] = np.mean(thetai[-5000:], axis=0)[:num_factors]

                    thetai = np.abs(np.dot(Jt, factors2))
                    theta2[m, :, i, j] = np.mean(thetai[-5000:], axis=0)[:num_factors]

            overlap_dict = {}
            overlap_dict['two_point_weights'] = two_point_weight_range
            overlap_dict['num_factors'] = num_factors
            overlap_dict['num_input_samples'] = num_input_samples
            overlap_dict['num_its_per_sample'] = num_its_per_sample
            overlap_dict['Nt'] = Nt
            overlap_dict['num_avg'] = 5000
            overlap_dict['two_point_overlap'] = theta1
            overlap_dict['three_point_overlap'] = theta2

            with open(os.path.join(data_dir, sim_file), 'wb') as handle:
                pickle.dump(overlap_dict, handle)

    two_pt_overlap = overlap_dict['two_point_overlap']
    three_pt_overlap = overlap_dict['three_point_overlap']

    two_pt_overlap_mean = np.nanmean(two_pt_overlap, axis=(2, 3))
    two_pt_overlap_std = np.nanstd(two_pt_overlap, axis=(2, 3)) / np.sqrt(two_pt_overlap.shape[2] * two_pt_overlap.shape[3])

    three_pt_overlap_mean = np.nanmean(three_pt_overlap, axis=(2, 3))
    three_pt_overlap_std = np.nanstd(three_pt_overlap, axis=(2, 3)) / np.sqrt(three_pt_overlap.shape[2] * three_pt_overlap.shape[3])

    two_pt_weights = overlap_dict['two_point_weights']

    ax9 = fig.add_subplot(2, 4, 4)
    ax10 = fig.add_subplot(2, 4, 8)

    for i in range(5):
        ax9.fill_between(two_pt_weights, two_pt_overlap_mean[:, i]+two_pt_overlap_std[:, i], two_pt_overlap_mean[:, i]-two_pt_overlap_std[:, i], color=colors[0], alpha=0.2/(1+i))
        ax9.plot(two_pt_weights, two_pt_overlap_mean[:, i], color=colors[0], linewidth=2, alpha=1/(1+i))

        ax10.fill_between(two_pt_weights, three_pt_overlap_mean[:, i]+three_pt_overlap_std[:, i], three_pt_overlap_mean[:, i]-three_pt_overlap_std[:, i], color=colors[1], alpha=0.2/(1+i))
        ax10.plot(two_pt_weights, three_pt_overlap_mean[:, i], color=colors[1], linewidth=2, alpha=1/(1+i))

    # ax9.imshow(two_pt_overlap, clim=(0, .8), cmap='cividis')
    # ax10.imshow(three_pt_overlap, clim=(0, .8), cmap='cividis')
    # ax10.set_xlabel('Factor')
    # ax10.set_ylabel('2-point weight')

    ### figure formatting

    ax1a.set_ylabel('Eigenvalue', fontsize=fontsize)
    ax1a.set_xlabel('Eigenvector', fontsize=fontsize)

    ax2a.set_xlabel("3-pt e'vec", fontsize=fontsize)
    ax2a.set_ylabel("2-pt e'vec", fontsize=fontsize)
    ax2a.set_title('Alignment', fontsize=fontsize)

    ax1a.set_xlim((0, 10))
    ax2a.set_xlim((-.5, 9.5))
    ax2a.set_ylim((-.5, 9.5))


    Nt = 15000
    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([0, Nt//2, Nt])

    for ax in [ax1]:
        ax.set_ylim((-.06, .06))

    for ax in [ax2, ax3]:#, ax9, ax10]:
        ax.set_ylim((0, 1))
        ax.set_yticks((0, 1))

    # for ax in [ax4, ax5, ax6]:
        # ax.set_yticklabels([])

    for ax in [ax1, ax2]:
        ax.set_xticklabels([])

    for ax in [ax7, ax8]:
        ax.set_yticks([])
        # ax.set_yticklabels([])

    # for ax in [ax4, ax5, ax6]:
    #     ax.set_yticklabels([])

    # ax7.set_xticklabels([])

    ax1.set_title(r'$A=({},{})$'.format(A1[0],A1[1]), fontsize=fontsize)
    # ax4.set_title('A=({},{})'.format(A2[0],A2[1]), fontsize=fontsize)
    ax3.set_xlabel('Time', fontsize=fontsize)
    # ax6.set_xlabel('Time', fontsize=fontsize)
    ax1.set_ylabel('Synaptic\nweight', fontsize=fontsize)

    ax2.set_ylabel('2-point\nalignment', fontsize=fontsize)
    ax3.set_ylabel('3-point\nalignment', fontsize=fontsize)

    ax7.set_title('2-point')
    ax8.set_title('3-point')
    # ax7.set_title('A=({},{})'.format(A1[0],A1[1]), fontsize=fontsize)
    # ax8.set_title('A=({},{})'.format(A2[0],A2[1]), fontsize=fontsize)

    ax8.set_xlabel('Alignment to \nfirst factor', fontsize=fontsize)
    ax8.set_ylabel('Frequency', fontsize=fontsize)
    ax7.legend(loc=0, frameon=False, fontsize=fontsize)

    ax10.set_xlabel('2-point \nweight, '+r'$A_1$')
    ax10.set_ylabel('3-point alignment')
    ax9.set_ylabel('2-point alignment')


    sns.despine(fig)
    fig.tight_layout()

    fig.savefig(savefile)


if __name__ == '__main__':

    plot_fig_natural_images_multi_term(Npix=35)

    # ### compute svd of input correlation tensors
    # num_input_samples = 10
    # Npix = 35
    # for i in range(num_input_samples):

    #     filename = 'inputs_factors_multiterm_Npix{}_iter{}.pkl'.format(Npix, i)
    #     _, _, _ = generate_inputs_compute_factors(xx=Npix, yy=Npix, savefile=filename)



    # if (sys.platform != 'darwin') and ('win' in sys.platform):
    #     datafile_head="C:/Users/Gabe/Dropbox (BOSTON UNIVERSITY)/Papers/tensor_hebb/Code"
    # else:
    #     datafile_head = "/Users/gabeo/Dropbox (BOSTON UNIVERSITY)/Papers/tensor_hebb/Code"

    # # run_asymmetry_error(Npix=35, rerun=True)

    # # run loop over initial conditions for a few parameter sets
    # loop_initial_cond(Npix=35, rerun=True, A=[-1, -1])
    # loop_initial_cond(Npix=35, rerun=True, A=[-1, .5])
    # loop_initial_cond(Npix=35, rerun=True, A=[-1, 1])


    # # plot_fig_2(Npix=35, Nplot=10, data_dir=datafile_head)