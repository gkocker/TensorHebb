import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import norm
from PIL import Image, ImageOps
from inequality.gini import Gini as gini
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import convolve2d
from sklearn import preprocessing
import tensortools as tt

import tensorly as tl
from tensorly.decomposition import tucker

fontsize = 10
labelsize = 8

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ['k']+colors

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    '''
    angle between v1 and v2 (radians)
    '''
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def generate_inputs_compute_factors(xx=10, yy=10, num_inputs=200, rank=20, a=2, b=1, savefile='inputs_factors.pkl'):


    results_dict = {}

    inputs = generate_inputs(xx=xx, yy=yy, num_inputs=num_inputs)

    if a == 3:
        mu = np.einsum('ij,ik,il,im',inputs**b,inputs,inputs,inputs) / num_inputs
    elif a == 2:
        mu = np.einsum('ij,ik,il',inputs**b,inputs,inputs) / num_inputs
    else:
        print('write input correlation computation')

    results_dict['inputs'] = inputs

    w, f = tucker(mu, rank=rank, verbose=True)
    
    f = f[0]
    if b > 1: 
        raise Warning('Computing the mode-0 factors of mu')
    
    lam = np.linalg.norm(tl.unfold(w, 0), axis=1)
    
    results_dict['singular_values'] = lam
    results_dict['singular_vectors'] = f

    with open(savefile, 'wb') as handle:
        pickle.dump(results_dict, handle)

    return inputs, lam, f


def generate_inputs(im_dir='/Users/gabeo/Documents/projects/images/berkeley_segmentation/train', xx=10, yy=10, num_inputs=200):

    im_files = os.listdir(im_dir)
    im_files = [f for f in im_files if f[-3:] == 'jpg']

    K = xx * yy
    inputs = []

    for i in range(num_inputs):

        im_file = np.random.choice(im_files)
    # for i, im_file in enumerate(im_files):

        x = Image.open(os.path.join(im_dir, im_file))
        x = ImageOps.grayscale(x)
        x = np.array(x)

        start_x = x.shape[0]//2 + np.random.choice(range(-x.shape[0]//2 + 1, x.shape[0]//2-xx))
        end_x = start_x + xx

        start_y = x.shape[1]//2 + np.random.choice(range(-x.shape[1]//2 + 1, x.shape[1]//2-yy))
        end_y = start_y + yy

        x_new = x[start_x:end_x, start_y:end_y].astype(np.float32)
    #     x_new = x_new - np.amin(x_new)
    #     x_new = x_new / np.amax(x_new)
    #     x_new = preprocessing.scale(x_new)
        x_new -= np.mean(x_new)
        x_new = preprocessing.normalize(x_new)# - np.mean(x_new), norm='l2')
        x_new -= np.mean(x_new)

        # x_new += np.std(x_new)/2.

        inputs.append(x_new)

    inputs = np.array(inputs).astype(np.float32)
    num_inputs = inputs.shape[0]
    inputs = inputs.reshape((num_inputs, K))
    
    return inputs


def run_sim(inputs, Nt=10000, eta=0.001, a=2, b=1, c=0, p=2):

    if len(inputs.shape) != 2:
        raise Exception('Input array should be 2d, is {}d'.format(len(inputs.shape)))

    num_inputs, K = inputs

    J = np.random.randn(K)
    J /= np.linalg.norm(J, ord=p)

    for t in range(1, Nt):
                    
        # draw input pattern
        x = inputs[np.random.choice(num_inputs)]
        
        # update weights
        n = J.dot(x)
        F = n**a * x**b * J**c
        
        dJ = F - J * (F * J * np.abs(J)**(p-2)).sum()
        J = J + eta * dJ
        
    #     negind = np.where(Jt[t] < 0)[0]
    #     Jt[t, negind] = 0.
        
        if np.any(~np.isfinite(J)):
            break
            
    print('stop at t={}/{}'.format(t+1, Nt))

    return J


def run_sim_return_traces(inputs, Nt=5000, eta=0.001, a=2, b=1, c=0, p=2, J0=None):

    if len(inputs.shape) != 2:
        raise Exception('Input array should be 2d, is {}d'.format(len(inputs.shape)))

    num_inputs, K = inputs.shape


    if J0 is None:
        J0 = np.random.randn(K)
    # J0 = np.ones(K,)
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
        F = n**a * x**b * J**c
        
        dJ = F - J * (F * J * np.abs(J)**(p-2)).sum()
        Jt[t] = J + eta * dJ
        
    #     negind = np.where(Jt[t] < 0)[0]
    #     Jt[t, negind] = 0.
        
        if np.any(~np.isfinite(Jt[t])):
            break
            
    print('stop at t={}/{}'.format(t+1, Nt))

    return Jt, x_t


def get_closest_factor(U, Ja, a, p=2):

    '''
    get the unit-norm CP factor closest to J
    '''

    factor_ind1 = np.argmax(Ja.dot(U.factors[0]))
    factor_ind2 = np.argmax(-Ja.dot(U.factors[0]))

    if Ja.dot(U.factors[0][:, factor_ind1]) > -Ja.dot(U.factors[0][:, factor_ind2]):
        factor = U.factors[0][:, factor_ind1]
    else:
        factor = -U.factors[0][:, factor_ind2]

    best_mode = 0
    for r in range(1, a+1):
        factor_ind1 = np.argmax(Ja.dot(U.factors[r]))
        factor_ind2 = np.argmax(-Ja.dot(U.factors[r]))

        if Ja.dot(U.factors[r][:, factor_ind1]) > -Ja.dot(U.factors[r][:, factor_ind2]):
            factor_tmp = U.factors[r][:, factor_ind1]
        else:
            factor_tmp = -U.factors[r][:, factor_ind2]

        if Ja.dot(factor_tmp) > Ja.dot(factor):
            factor = factor_tmp
            best_mode = r

    return factor / np.linalg.norm(factor, ord=p)


def loop_initial_cond(Nt=15000, Ninit=20, Ntsave=2000, Nsave=100, eta=0.001, a=2, b=1, c=0, p=2, Npix=10, datafile_head='fig1_data_orth_error', rerun=False):

    datafile = datafile_head+'_a={}_b={}_c={}.pkl'.format(a, b, c)
    if os.path.exists(datafile):
        print('Model ensemble already exists at {}'.format(datafile))
        
        if not rerun:
            results_dict = pickle.load(datafile)
            return results_dict['angle_factor_weights']

        else: os.remove(datafile)

    else: pass

    if Nsave > Npix**2:
        Nsave = Npix**2
    else: pass

    if Nt % Ntsave != 0:
        raise Exception('Number of save points needs to evenly divide the total time')

    theta = np.zeros((Nt, Ninit))
    factors = np.zeros((Npix**2, Ninit))
    Jsave = np.zeros((Ntsave, Nsave, Ninit))
    indices = np.zeros((Nsave, Ninit))

    tsave = range(0, Nt, Nt//Ntsave)

    results_dict = {}
    results_dict['a'] = a
    results_dict['b'] = b
    results_dict['c'] = c
    results_dict['p'] = p
    results_dict['Npix'] = Npix
    results_dict['tsave'] = tsave

    for n in range(Ninit):

        inputs = generate_inputs(xx=Npix, yy=Npix)
        num_inputs, K = inputs.shape

        if a == 3:
            mu = np.einsum('ij,ik,il,im',inputs**b,inputs,inputs,inputs) / num_inputs
        elif a == 2:
            mu = np.einsum('ij,ik,il',inputs**b,inputs,inputs) / num_inputs
        else:
            print('write input correlation computation')

        Jt, x_t = run_sim_return_traces(inputs, Nt=Nt, eta=eta, a=a, b=b, c=c, p=p)

        ind_save = np.random.choice(Npix**2, size=Nsave, replace=False)
        indices[:, n] = ind_save
        Jsave[:, :, n] = Jt[tsave][:,ind_save]

        Jplot = np.mean(Jt[-1000:, :], axis=0)
        U = tt.cp_als(mu, rank=1, verbose=False)
        factor = get_closest_factor(U, Jplot, a)

        for t in range(Nt):
            theta[t, n] = angle_between(Jt[t], factor)

        factors[:, n] = factor
    
    results_dict['factors'] = factors
    results_dict['angle_factor_weights'] = theta
    results_dict['weights_over_time'] = Jsave
    results_dict['weight_indices'] = indices

    with open(datafile, 'wb') as handle:
        pickle.dump(results_dict, handle)

    del mu

    return theta


def plot_image_stack(ax, data, Nplot=4, cmap='cividis', cscale=.1):
    
    if len(data.shape) != 3:
        raise Exception('need inputs in shape (num_inputs, x, y)')
    
    num_inputs, X, Y = data.shape
    Xmesh, Ymesh = np.meshgrid(np.arange(X), np.arange(Y))
    
    plot_ind = list(range(0, num_inputs, num_inputs//Nplot))
    if len(plot_ind) > Nplot:
        plot_ind = plot_ind[:Nplot]
    
    cmid = np.mean(data)
    cmin = cmid - (np.amax(data) - np.amin(data))*cscale
    cmax = cmid + (np.amax(data) - np.amin(data))*cscale

    scam = plt.cm.ScalarMappable(
        norm=plt.cm.colors.Normalize(cmin, cmax),
        cmap=cmap
    )

    for n, z in enumerate(plot_ind):
        Zmesh = z * np.ones((X, Y))
        
        scam.set_array([])   
        surf = ax.plot_surface(
            Xmesh, Ymesh, Zmesh,
            facecolors  = scam.to_rgba(data[z]),
            antialiased = True,
            rstride=1, cstride=1, alpha=None
        )

    return scam, plot_ind


def plot_3pt_corr(ax, inputs, b=1, Nplot=4, Kplot=100, Nsmooth=31, cmap='cividis', cscale=.35):

    num_inputs, K = inputs.shape
    # xplot = range(0, K, K//Kplot)
    # inputs = inputs.T[xplot].T
    # _, K = inputs.shape

    print(K)
    plot_ind = list(range(0, K, K//(Nplot)))
    if len(plot_ind) == Nplot+1:
        plot_ind = plot_ind[:-1]

    print(len(plot_ind))
    if K % Nsmooth == 0:
        mu = np.zeros((K//(Nsmooth//2), K//(Nsmooth//2), Nplot))
    else:
         mu = np.zeros((K//(Nsmooth//2)+1, K//(Nsmooth//2)+1, Nplot))
    inputsb = inputs**b

    for i, z in enumerate(plot_ind):
        inputs2 = inputs * np.outer(inputs[:, z], np.ones(K,))
        # mu[:, :, i] = inputsb.T.dot(inputs2) / num_inputs
        # mu[:, :, i] = np.einsum('ij,ik',inputs**b,inputs * np.outer(inputs[:, z], np.ones(K,))) / num_inputs

        ### smoothed 3pt corr
        mui = inputsb.T.dot(inputs2) / num_inputs
        mui = convolve2d(mui, np.ones((Nsmooth, Nsmooth)), mode='same', boundary='wrap')
        mui = mui[::Nsmooth//2, ::Nsmooth//2]
        mu[:, :, i] = mui

    Xmesh, Ymesh = np.meshgrid(np.arange(mu.shape[0]), np.arange(mu.shape[1]))

    cmid = np.median(mu)
    crange = np.amax(mu) - np.amin(mu)
    cmin = cmid - cscale*crange
    cmax = cmid + cscale*crange

    scam = plt.cm.ScalarMappable(
        norm=plt.cm.colors.Normalize(cmin, cmax),
        cmap=cmap
    )

    for z in range(Nplot):
        Zmesh = z * np.ones((mu.shape[0], mu.shape[1]))
        scam.set_array([])   
        surf = ax.plot_surface(
            Xmesh, Ymesh, Zmesh,
            facecolors  = scam.to_rgba(mu[:, :, z]),
            antialiased = True,
            rstride=1, cstride=1, alpha=None
        )

    return scam, plot_ind


def plot_orthog_approx_error(ax, results_dict, color='k', jitter=0.1, label=None):

    marker_style = {'s':20, 'c':color}
    line_style = {'linewidth':3,'color':color}

    ranks = results_dict['ranks']
    # for i, r in enumerate(results_dict['ranks']):
    #     err = results_dict['objectives'][i]
    #     nscat = len(err)
    #     x = r * np.ones(nscat) + (np.random.rand(nscat)-0.5)*jitter

    #     ax.scatter(x, err, **marker_style)

    err = np.array(results_dict['objectives']).mean(axis=1)
    err_std = np.array(results_dict['objectives']).std(axis=1)

    ax.fill_between(ranks, err - err_std, err+err_std, alpha=0.2, color=color)
    ax.plot(results_dict['ranks'], err, **line_style, label=label)

    return None


def run_orthog_approx_error(a=2, b=1, max_rank=4, Npix=10, replicates=4, datafile_head='fig1_data_ensemble', rerun=False):

    datafile = datafile_head+'_a={}_b={}.pkl'.format(a, b)
    if os.path.exists(datafile):
        print('Model ensemble already exists at {}'.format(datafile))
        
        if not rerun:
            results_dict = pickle.load(datafile)
            return results_dict['objectives']

        else: pass

    else: pass

    inputs = generate_inputs(xx=Npix, yy=Npix)
    num_inputs = inputs.shape[0]
    ranks = range(1, max_rank)

    if a == 2:
        mu = np.einsum('ij,ik,il',inputs**b,inputs,inputs) / num_inputs
    elif a == 1:
        mu = np.einsum('ij,ik',inputs**b,inputs) / num_inputs
    else:
        raise Exception('Compute input correlation for a={}'.format(a))

    ensemble = tt.Ensemble(fit_method="cp_als")
    ensemble.fit(mu, ranks=ranks, replicates=replicates)

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

    del mu
    del ensemble

    return objectives


def plot_fig_1(savefile='fig1.pdf', datafile_head='fig1_data', cmap='cividis', max_rank=4, Npix=10, Ninit=20, Nsmooth=30, cscale=0.35, data_dir="/Users/gabeo/Dropbox (BOSTON UNIVERSITY)/Papers/tensor_hebb/Code"):

    fig = plt.figure(figsize=(5.67, 3.7))

    ### plot example image patches
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')

    inputs = generate_inputs(xx=Npix, yy=Npix)
    num_inputs = inputs.shape[0]
    scam, _ = plot_image_stack(ax1, data=inputs.reshape((num_inputs, Npix, Npix)), cscale=.1, cmap=cmap)
    # ax1.set_xticks(range(Npix, 2))
    # ax1.set_yticks(range(Npix, 2))
    # ax1.set_zticks(range(Npix, 2))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])

    # ax1.set_xticklabels(range(Npix, 2), fontsize=labelsize)
    # ax1.set_yticklabels(range(Npix, 2), fontsize=labelsize)
    # ax1.set_zticklabels(range(Npix, 2), fontsize=labelsize)

    ### plot input correlation mu for a=1, b=2
    a=1
    b=2
    num_inputs, K = inputs.shape
    mu = np.einsum('ij,ik',inputs**b,inputs) / num_inputs

    mu = convolve2d(mu, np.ones((Nsmooth, Nsmooth)), mode='same', boundary='wrap')
    mu = mu[::Nsmooth//2, ::Nsmooth//2]

    cmid = np.median(mu)
    crange = np.amax(mu) - np.amin(mu)
    clim = (cmid - cscale*crange, cmid + cscale*crange)

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(mu, cmap=cmap, clim=clim)
    ax2.set_xlabel('Pixel', fontsize=fontsize)
    ax2.set_ylabel('Pixel', fontsize=fontsize)
    ax2.set_title('3-point corr.\n(a,b)=(1,{})'.format(b), fontsize=fontsize)

    # ax2.set_xticks(range(Npix, 2))
    # ax2.set_yticks(range(Npix, 2))
    # ax2.set_yticks(range(0, K//(Nsmooth//2), K//(Nsmooth//4)))
    # ax2.set_xticks(range(0, K//(Nsmooth//2), K//(Nsmooth//4)))
    # ax2.set_xticklabels(range(0, Npix, Npix//2), fontsize=labelsize)
    # ax2.set_yticklabels(range(0, Npix, Npix//2), fontsize=labelsize)

    # results_dict['panel2_corr_a=1_b=2'] = mu

    ### plot input correlation for a=2, b=1
    a=2
    b=1

    # mu = np.einsum('ij,ik,il',inputs**b,inputs,inputs) / num_inputs
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    scam, plot_ind = plot_3pt_corr(ax3, inputs=inputs, b=b, cscale=0.25, cmap=cmap, Nsmooth=Nsmooth)

    ax3.set_title('3-point corr.\n(a,b)=(2,{})'.format(b), fontsize=fontsize)

    # ax3.set_xticks(range(Npix, 2))
    # ax3.set_yticks(range(Npix, 2))
    # ax3.set_zticks(range(len(plot_ind), 2))
    # ax3.set_zticklabels(plot_ind, fontsize=labelsize)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    ax3.set_xlabel('Pixel', fontsize=fontsize)
    ax3.set_ylabel('Pixel', fontsize=fontsize)
    ax3.set_zlabel('Pixel', fontsize=fontsize)

    # results_dict = {}
    # results_dict['panel3_corr_a=2_b=1'] = mu[:, :, plot_ind]
    # datafile = datafile_head+'_panel3_corr.pkl'
    # with open(datafile, 'wb') as handle:
    #     pickle.dump(results_dict, handle)

    ### plot error of orthogonal approximation (CP decomposition) for (a, b)=(2, 1) and (2, 2)
    ax4 = fig.add_subplot(2, 3, 4)
    datafile = os.path.join(data_dir, 'fig1_data_ensemble_a=2_b=1.pkl')

    with open(datafile, 'rb') as handle:
        results_dict = pickle.load(handle)

    _ = plot_orthog_approx_error(ax4, results_dict, color=colors[0], label='(a,b)=(2,1)')

    datafile = os.path.join(data_dir, 'fig1_data_ensemble_a=2_b=2.pkl')
    with open(datafile, 'rb') as handle:
        results_dict = pickle.load(handle)

    _ = plot_orthog_approx_error(ax4, results_dict, color=colors[1], label='(a,b)=(2,2)')

    ax4.set_xlabel('Rank of orth.\napproximation', fontsize=fontsize)
    ax4.set_ylabel('MSE of orth.\napproximation', fontsize=fontsize)
    ax4.legend(loc=0, frameon=False, fontsize=fontsize)
    ax4.set_ylim((0, ax4.get_ylim()[1]))

    ### learning dynamics for a=2, b=1
    a = 2
    b = 1
    num_inputs, K = inputs.shape

    Jt, _ = run_sim_return_traces(inputs=inputs, a=a, b=b, Nt=15000)

    ax5 = fig.add_subplot(2, 3, 5)
    plot_ind = np.random.choice(K, 5)
    ax5.plot(Jt[:, plot_ind], linewidth=2)

    ax5.set_xlabel('Time', fontsize=fontsize)
    ax5.set_ylabel('Synaptic weight', fontsize=fontsize)

    ### angle between weight vector and singular vector 
    ax6 = fig.add_subplot(2, 3, 6)

    # T = 15000
    Ntplot = 200
    # tplot = range(0, T, T//Ntplot)
    datafile = os.path.join(data_dir, 'fig1_data_orth_error_a=2_b=1_c=0.pkl')
    with open(datafile, 'rb') as handle:
        results_dict = pickle.load(handle)

    error_angle = results_dict['angle_factor_weights']

    T = error_angle.shape[0]
    tplot = range(0, T, T//Ntplot)

    # error_angle = loop_initial_cond(a=2, b=1, c=0, p=2, Ninit=Ninit, Npix=Npix, Nt=tplot)
    # results_dict['panel6_error_angle_a=2_b=1'] = error_angle

    error_std = np.nanstd(error_angle, axis=1) / np.sqrt(Ninit)
    error_mean = np.nanmean(error_angle, axis=1)
    error_std = error_std[tplot]
    error_mean = error_mean[tplot]

    ax6.fill_between(tplot, error_mean + error_std, error_mean - error_std, alpha=0.2, color=colors[0])
    ax6.plot(tplot, error_mean, linewidth=2, color=colors[0], label='(a,b)=(2,1)')

    datafile = os.path.join(data_dir, 'fig1_data_orth_error_a=2_b=2_c=0.pkl')
    with open(datafile, 'rb') as handle:
        results_dict = pickle.load(handle)

    error_angle = results_dict['angle_factor_weights']
    # error_angle = loop_initial_cond(a=2, b=1, c=0, p=2, Ninit=Ninit, Npix=Npix, Nt=tplot)
    # results_dict['panel6_error_angle_a=2_b=1'] = error_angle

    error_std = np.nanstd(error_angle, axis=1) / np.sqrt(Ninit)
    error_mean = np.nanmean(error_angle, axis=1)
    error_std = error_std[tplot]
    error_mean = error_mean[tplot]

    ax6.fill_between(tplot, error_mean + error_std, error_mean - error_std, alpha=0.2, color=colors[1])
    ax6.plot(tplot, error_mean, linewidth=2, color=colors[1], label='(a,b)=(2,2)')

    ax6.set_xlabel('Time', fontsize=fontsize)
    ax6.set_ylabel('Angle between J and u', fontsize=fontsize)

    ax6.set_ylim((-.1, np.pi/2+.1))
    ax6.set_yticks((0, np.pi/4, np.pi/2))
    ax6.set_yticklabels(['0', r'$\pi/4$', r'$\pi/2$'], fontsize=labelsize)
    ax6.legend(loc=0, frameon=False, fontsize=fontsize)

    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(savefile)

    # with open(datafile, 'wb') as handle:
    #     pickle.dump(results_dict, handle)


def plot_fig_natural_images(savefile='fig_natim.pdf', datafile_head='fig1_data', cmap='cividis', max_rank=4, Npix=10, Ninit=20, Nsmooth=30, cscale=0.35, data_dir="/Users/gabeo/Dropbox (BOSTON UNIVERSITY)/Papers/tensor_hebb/Code"):

    fig = plt.figure(figsize=(5.67, 6)) # 5.7 height for 2 rows

    ### plot example image patches
    ax1 = fig.add_subplot(3, 3, 1, projection='3d')

    inputs = generate_inputs(xx=Npix, yy=Npix)
    num_inputs = inputs.shape[0]
    scam, _ = plot_image_stack(ax1, data=inputs.reshape((num_inputs, Npix, Npix)), cscale=.1, cmap=cmap)
    # ax1.set_xticks(range(Npix, 2))
    # ax1.set_yticks(range(Npix, 2))
    # ax1.set_zticks(range(Npix, 2))
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_zticks([])


    ### plot input correlation for a=1, b=2
    a = 1
    b = 2

    num_inputs, K = inputs.shape
    mu = np.einsum('ij,ik',inputs**b,inputs) / num_inputs

    mu = convolve2d(mu, np.ones((Nsmooth, Nsmooth)), mode='same', boundary='wrap')
    mu = mu[::Nsmooth//2, ::Nsmooth//2]

    cmid = np.median(mu)
    crange = np.amax(mu) - np.amin(mu)
    clim = (cmid - cscale*crange, cmid + cscale*crange)

    ax2 = fig.add_subplot(3, 3, 2)
    ax2.imshow(mu, cmap=cmap, clim=clim)

    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2.set_xlabel('Input', fontsize=fontsize)
    ax2.set_ylabel('Input', fontsize=fontsize)
    ax2.set_title('3-point corr.\n(a,b)=(1,{})'.format(b), fontsize=fontsize)


    ### plot input correlation for a=2, b=1
    a=2
    b=1

    # mu = np.einsum('ij,ik,il',inputs**b,inputs,inputs) / num_inputs
    ax3 = fig.add_subplot(3, 3, 3, projection='3d')
    scam, plot_ind = plot_3pt_corr(ax3, inputs=inputs, b=b, cscale=0.25, cmap=cmap, Nsmooth=Nsmooth)

    ax3.set_title('3-point corr.\n(a,b)=(2,{})'.format(b), fontsize=fontsize)

    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_zticks([])
    ax3.set_xlabel('Input', fontsize=fontsize)
    # ax3.set_ylabel('Input', fontsize=fontsize)
    ax3.set_zlabel('Input', fontsize=fontsize)

    ### plot error of orthogonal approximation (CP decomposition) for (a, b)=(2, 1) and (2, 2)
    ax4 = fig.add_subplot(3, 3, 4)
    datafile = os.path.join(data_dir, 'fig1_data_ensemble_a=2_b=1.pkl')

    with open(datafile, 'rb') as handle:
        results_dict = pickle.load(handle)

    _ = plot_orthog_approx_error(ax4, results_dict, color=colors[0], label='(a,b)=(2,1)')

    datafile = os.path.join(data_dir, 'fig1_data_ensemble_a=2_b=2.pkl')
    with open(datafile, 'rb') as handle:
        results_dict = pickle.load(handle)

    _ = plot_orthog_approx_error(ax4, results_dict, color=colors[1], label='(a,b)=(2,2)')

    ax4.set_xlabel('Rank of orth.\napproximation', fontsize=fontsize)
    ax4.set_ylabel('MSE of orth.\napproximation', fontsize=fontsize)
    ax4.legend(loc=0, frameon=False, fontsize=fontsize)
    ax4.set_ylim((0, ax4.get_ylim()[1]))

    ### learning dynamics for a=2, b=1
    a = 2
    b = 1
    dt_ds = 100
    Nt = 15000
    num_inputs, K = inputs.shape

    Jt, _ = run_sim_return_traces(inputs=inputs, a=a, b=b, Nt=Nt)
    tplot = np.arange(0, Nt, dt_ds)

    ax5 = fig.add_subplot(3, 3, 5)
    plot_ind = np.random.choice(K, 5)

    for i in plot_ind:
        Jt_i = Jt[:, i]
        Jt_i_plot = running_mean(Jt_i, N=dt_ds)
        ax5.plot(tplot, Jt_i_plot[::dt_ds], linewidth=1, color='k')

    # ax5.set_xlabel('Time', fontsize=fontsize)
    ax5.set_ylabel('Synaptic\nweight', fontsize=fontsize)

    ### overlap between weight vector and singular vector 
    ax6 = fig.add_subplot(3, 3, 8)

    num_its_per_sample = 100
    Nt = 40000
    dt_ds = 100
    eta = .001
    a = 2
    b = 1
    c = 0
    p = 2

    input_files = [f for f in os.listdir(data_dir) if (f.split('_')[0] == 'inputs')]

    sim_file = 'inputs_factors_Npix35_a2_sims.pkl'
    input_files.remove(sim_file)

    multi_term_input_files = [f for f in input_files if ('multiterm' in f)]
    for f in multi_term_input_files:
        input_files.remove(f)

    num_input_samples = len(input_files)

    if os.path.exists(os.path.join(data_dir, sim_file)):
        with open(os.path.join(data_dir, sim_file), 'rb') as handle:
            overlap_dict = pickle.load(handle)

    else:
        print('Running sims for factor overlap')

        theta = np.zeros((Nt//dt_ds, num_input_samples, num_its_per_sample))
        factor_num = np.zeros((num_input_samples, num_its_per_sample))

        for i, f in enumerate(input_files):
            print('{}/{}'.format(i, len(num_input_samples)))

            with open(os.path.join(data_dir, f), 'rb') as handle:
                input_file = pickle.load(handle)

            inputs = input_file['inputs']
            factors = input_file['singular_vectors']

            for j in range(num_its_per_sample):
                Jt, _ = run_sim_return_traces(inputs, a=a, b=b, c=c, p=p, Nt=Nt, eta=eta)
                Jt = Jt[::dt_ds]
                
                thetai = np.abs(np.dot(Jt, factors))
                fac_ind = np.argmax(np.mean(thetai[-10:], axis=0))
                
                theta[:, i, j] = thetai[:, fac_ind]
                factor_num[i, j] = fac_ind

        overlap_dict = {}
        overlap_dict['overlap'] = theta
        overlap_dict['factor_number'] = factor_num

        with open(os.path.join(data_dir, sim_file), 'wb') as handle:
            pickle.dump(overlap_dict, handle)

    theta = overlap_dict['overlap']
    factor_num = overlap_dict['factor_number']    

    print(theta.shape)
    print(factor_num.shape)

    tplot = range(0, Nt, dt_ds)
    
    theta_m = np.mean(theta, axis=(1, 2))
    theta_s = np.std(theta, axis=(1, 2)) / np.sqrt(num_its_per_sample * num_input_samples)

    ax6.fill_between(tplot, theta_m-theta_s, theta_m+theta_s, color='k', alpha=0.2)
    ax6.plot(tplot, theta_m, 'k', linewidth=2)

    ax6.set_ylim((0, 1.1))
    ax6.set_xlabel('Time', fontsize=fontsize)
    ax6.set_ylabel('Overlap', fontsize=fontsize)

    ax7 = fig.add_subplot(3, 3, 7)
    ax7.hist(factor_num.reshape(-1,), bins=range(0, 11), align='left', color='k', density=True)
    ax7.set_xlabel('Singular vector')
    ax7.set_ylabel('Frequency')


    ### plot example final components

    f = input_files[0]
    with open(os.path.join(data_dir, f), 'rb') as handle:
        input_file = pickle.load(handle)

    inputs = input_file['inputs']
    factors = input_file['singular_vectors']

    Jt, _ = run_sim_return_traces(inputs, a=a, b=b, c=c, p=p, Nt=Nt, eta=eta)
    Jplot = np.mean(Jt[-100:], axis=0)

    thetai = np.abs(np.dot(Jplot, factors))
    fac_ind = np.argmax(thetai)
    Jplot = Jplot.reshape((Npix, Npix))

    ax8 = fig.add_subplot(3, 3, 6)
    ax8.imshow(Jplot, cmap='cividis')
    ax8.set_xticks([])
    ax8.set_yticks([])
    ax8.set_title('closest factor: {}'.format(fac_ind), fontsize=fontsize)

    f = input_files[1]
    with open(os.path.join(data_dir, f), 'rb') as handle:
        input_file = pickle.load(handle)

    inputs = input_file['inputs']
    factors = input_file['singular_vectors']

    Jt, _ = run_sim_return_traces(inputs, a=a, b=b, c=c, p=p, Nt=Nt, eta=eta)
    Jplot = np.mean(Jt[-100:], axis=0)

    thetai = np.abs(np.dot(Jplot, factors))
    fac_ind = np.argmax(thetai)
    Jplot = Jplot.reshape((Npix, Npix))

    ax9 = fig.add_subplot(3, 3, 9)
    ax9.imshow(Jplot, cmap='cividis')
    ax9.set_xticks([])
    ax9.set_yticks([])
    ax9.set_title('closest factor: {}'.format(fac_ind), fontsize=fontsize)


    # ax6.set_ylim((-.1, np.pi/2+.1))
    # ax6.set_yticks((0, np.pi/4, np.pi/2))
    # ax6.set_yticklabels(['0', r'$\pi/4$', r'$\pi/2$'], fontsize=labelsize)
    # ax6.legend(loc=0, frameon=False, fontsize=fontsize)

    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(savefile)

    # with open(datafile, 'wb') as handle:
    #     pickle.dump(results_dict, handle)


if __name__ == '__main__':
    
    plot_fig_natural_images(Npix=35)
    
    # plot_fig_1(Npix=35, max_rank=30)

    # objectives = run_orthog_approx_error(a=2, b=1, max_rank=30, Npix=35, replicates=4, rerun=True)
    # objectives = run_orthog_approx_error(a=2, b=2, max_rank=30, Npix=35, replicates=4, rerun=True)

    # error_angle = loop_initial_cond(a=2, b=1, c=0, p=2, Ninit=20, Npix=35, Nt=500000, rerun=True)
    # error_angle = loop_initial_cond(a=2, b=2, c=0, p=2, Ninit=20, Npix=35, Nt=500000, rerun=True)


    ### compute svd of input correlation tensors
    # num_input_samples = 10
    # Npix = 35
    # for i in range(num_input_samples):

    #     a = 2
    #     filename = 'inputs_factors_Npix{}_a{}_iter{}.pkl'.format(Npix, a, i)
    #     _, _, _ = generate_inputs_compute_factors(xx=Npix, yy=Npix, a=a, savefile=filename)