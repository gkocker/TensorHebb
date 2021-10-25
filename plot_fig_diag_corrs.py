import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
# import SeabornFig2Grid as sfg
from scipy.linalg import norm
from PIL import Image, ImageOps
from inequality.gini import Gini as gini
from scipy.optimize import minimize, minimize_scalar
from sklearn import preprocessing
import tensortools as tt

fontsize = 10
labelsize = 8

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ['k']+colors


def run_sim(K=10, Nt=5000, eta=0.01, a=2, b=1, c=1, p=2, flat_init=False):

    if flat_init:
        J = np.ones(K,)
    else:
        J = np.random.randn(K)

    J /= np.linalg.norm(J, ord=p)

    for t in range(1, Nt):
                    
        # draw input pattern
        x = np.zeros(K,)
        xind = np.random.choice(K)
        x[xind] = np.random.normal(loc=0, scale=1)
        
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


def run_sim_return_traces(K=10, Nt=5000, eta=0.01, a=2, b=1, c=1, p=2, flat_init=False, norm_init_factor=1):

    Jt = np.zeros((Nt, K))

    if flat_init:
        J0 = np.ones(K,)
    else:
        J0 = np.random.randn(K,)

    J0 /= np.linalg.norm(J0, ord=p)*norm_init_factor

    Jt[0] = J0

    for t in range(1, Nt):
            
        J = Jt[t-1]
        
        # draw input pattern
        x = np.zeros(K,)
        xind = np.random.choice(K)
        x[xind] = np.random.normal(loc=0, scale=1)
        
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

    return Jt


def loop_initial_cond_lp(Ninit=20, K=10, Nt=5000, eta=0.01, a=2, b=1, c=1, p=2, flat_init=False, thresh=0.1, p_calc=0, norm_init_factor=1):

    n = 0
    lp_norm = np.zeros((Nt, Ninit))

    while n < Ninit:

        Jt = run_sim_return_traces(a=a, b=b, c=c, p=p, eta=eta, Nt=Nt, K=K, flat_init=flat_init, norm_init_factor=norm_init_factor)

        if ~np.any(~np.isfinite(Jt)):

            if p_calc == 0:
                lp_norm[:, n] = np.sum(np.abs(Jt) > thresh, axis=1)
            else:
                lp_norm[:, n] = np.linalg.norm(Jt, axis=1, ord=p_calc)

            n += 1

        else:
            pass
        
    return lp_norm


def final_weight_dist(K=10, Nt=5000, eta=0.01, a=2, b=1, c=1, p=2, Ninit=100, flat_init=False, Nsmooth=500):

    Jfinal = np.zeros((Ninit, K))

    n = 0

    while n < Ninit:

        # J = run_sim(a=a, b=b, c=c, p=p, eta=eta, Nt=Nt, K=K, flat_init=flat_init)
        Jt = run_sim_return_traces(K=K, Nt=Nt, a=a, b=b, c=c, p=p, eta=eta, flat_init=flat_init)
        if np.any(~np.isfinite(Jt)):
            pass
        else:
            # J = np.mean(Jt[-Nsmooth:], axis=0)
            J = Jt[-1]
            Jfinal[n] = J
            n += 1

    # for n in range(Ninit):
    #     Jfinal[n] = run_sim(a=a, b=b, c=c, p=p, eta=eta, Nt=Nt, K=K, flat_init=flat_init)

    return Jfinal


def plot_example_and_hist(a=2, b=1, c=1, K=10, Nplot=10, Ninit=20, Nt_ex=60000, Nt=100000, eta=.01, p=2, flat_init=False):

    tplot = range(Nt_ex)

    Jt = run_sim_return_traces(a=a, b=b, c=c, p=p, K=K, Nt=Nt_ex, eta=eta, flat_init=flat_init)
    plot_ind = np.random.choice(K, Nplot, replace=False)
    Jfinal = final_weight_dist(K=K, Ninit=Ninit, Nt=Nt, eta=eta, a=a, b=b, c=c, p=p, flat_init=flat_init)

    

    ### make the plot
    g = sns.JointGrid(tplot, Jt[:, plot_ind[0]], ratio=3)
    for i in range(1, Nplot):
        g.ax_joint.plot(tplot, Jt[:, plot_ind[i]], c=colors[i])

    plt.sca(g.ax_marg_y)
    sns.distplot(Jfinal.reshape(-1,), kde=False, vertical=True, color='k', norm_hist=True)
    # plt.xscale('log')

    ylim = g.ax_joint.get_ylim()
    g.ax_marg_y.set_ylim(ylim)

    g.ax_marg_x.remove()

    return g


def plot_fig_3(savefile='fig3.pdf', camp='cividis', K=10, Ninit=20, Nplot=None):
    '''
    plot figure for convergence to Lp sphere with a+c=1
    '''

    if (Nplot is None) or (Nplot > K):
        Nplot = K

    fig, ax = plt.subplots(2, 2, figsize=(3.5, 3.5))

    a = 1
    b = 1
    c = 0
    p = 2
    eta = .01
    Nplot = K

    Nt_ex=3000
    Ntplot = 200
    tplot = range(0, Nt_ex, Nt_ex//Ntplot)

    print('Running sims for c={}'.format(c))
    Jt = run_sim_return_traces(K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False)
    
    ax[0, 0].plot(Jt, linewidth=2)

    p_plot = [1, 2]
    norm_init_factors = [.5, 1, 2]

    for i, p in enumerate(p_plot):
        for j, norm_init in enumerate(norm_init_factors):

            lp_norm = loop_initial_cond_lp(Ninit=Ninit, K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False, p_calc=p, norm_init_factor=norm_init)
            plot_std = np.nanstd(lp_norm, axis=1) / np.sqrt(Ninit)
            plot_mean = np.nanmean(lp_norm, axis=1)

            plot_std = plot_std[::Nt_ex//Ntplot]
            plot_mean = plot_mean[::Nt_ex//Ntplot]

            ax[1, 0].fill_between(tplot, plot_mean + plot_std, plot_mean - plot_std, alpha=0.2, color=colors[i])
            if j == 0:
                if i == 0:
                    ax[1, 0].plot(tplot, plot_mean, linewidth=2, color=colors[i], label='p={}'.format(p))
                else:
                    ax[1, 0].plot(tplot, plot_mean, '--', linewidth=2, color=colors[i], label='p={}'.format(p))
            else:
                if i == 0:
                    ax[1, 0].plot(tplot, plot_mean, linewidth=2, color=colors[i])
                else:
                    ax[1, 0].plot(tplot, plot_mean, '--', linewidth=2, color=colors[i])


    ax[1, 0].legend(loc=0, frameon=False, fontsize=fontsize)
    
    ax[0, 0].set_xticklabels([])
    ax[0, 0].set_ylim((-1.1, 1.1))
    # ax[1, 0].set_ylim((0, 2))
    # ax[1, 0].set_yticks(np.linspace(1, K, 4, dtype=int))
    ax[0, 0].set_title('(a,b,c)=({},{},{})'.format(a,b,c), fontsize=fontsize)
    ax[1, 0].set_xlabel('Time')

    
    a = 2
    b = 1
    c = -1

    print('Running sims for c={}'.format(c))
    need_sim = True

    while need_sim:
        Jt = run_sim_return_traces(K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False)
        if ~np.any(~np.isfinite(Jt)):
            need_sim = False
        else:
            pass

    ax[0, 1].plot(Jt, linewidth=2)

    p_plot = [1, 2]
    norm_init_factors = [.5, 1, 2]

    for i, p in enumerate(p_plot):
        for j, norm_init in enumerate(norm_init_factors):

            lp_norm = loop_initial_cond_lp(Ninit=Ninit, K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False, p_calc=p, norm_init_factor=norm_init)
            plot_std = np.nanstd(lp_norm, axis=1) / np.sqrt(Ninit)
            plot_mean = np.nanmean(lp_norm, axis=1)

            plot_std = plot_std[::Nt_ex//Ntplot]
            plot_mean = plot_mean[::Nt_ex//Ntplot]

            ax[1, 1].fill_between(tplot, plot_mean + plot_std, plot_mean - plot_std, alpha=0.2, color=colors[i])
            if j == 0:
                if i == 0:
                    ax[1, 1].plot(tplot, plot_mean, linewidth=2, color=colors[i], label='p={}'.format(p))
                else:
                    ax[1, 1].plot(tplot, plot_mean, '--', linewidth=2, color=colors[i], label='p={}'.format(p))
            else:
                if i == 0:
                    ax[1, 1].plot(tplot, plot_mean, linewidth=2, color=colors[i])
                else:
                    ax[1, 1].plot(tplot, plot_mean, '--', linewidth=2, color=colors[i])

    ax[1, 1].legend(loc=0, frameon=False, fontsize=fontsize)
    
    ax[0, 1].set_xticklabels([])
    ax[0, 1].set_ylim((-1.1, 1.1))
    # ax[1, 0].set_ylim((0, 2))
    # ax[1, 0].set_yticks(np.linspace(1, K, 4, dtype=int))
    ax[0, 1].set_title('(a,b,c)=({},{},{})'.format(a,b,c), fontsize=fontsize)
    ax[1, 1].set_xlabel('Time')

    # ax[1, 0].set_ylim((0.75, 2.25))
    # ax[1, 1].set_ylim((0.75, 2.25))

    ax[0, 0].set_ylabel('Synaptic weight', fontsize=fontsize)
    ax[1, 0].set_ylabel('Synaptic weight norm', fontsize=fontsize)

    # ax[0, 1].set_ylabel('Synaptic weight', fontsize=fontsize)
    # ax[1, 1].set_ylabel('Nonzero synapses', fontsize=fontsize)

    # ymax = np.round(ax[1, 1].get_ylim()[1])
    # ax[1, 1].set_yticks(np.linspace(1, ymax, 4, dtype=int))
    # ax[1, 1].set_yticks(range(1, 9, 2))

    sns.despine(fig)
    fig.tight_layout()
    # gs.tight_layout(fig)

    fig.savefig(savefile)


def plot_fig_4(savefile='fig4.pdf', cmap='cividis', K=10, Ninit=20, Nplot=10, thresh=0.1):

    '''
    convergence to partially sparse solutions
    '''

    fig, ax = plt.subplots(2, 4, figsize=(7, 3.5))

    ### first (a, b, c) = (2, 1, 1) - get an example converging to the sparse solution at 1 or -1
    a = 2
    b = 1
    c = 1
    p = 2
    eta = .01
    Nplot = 10

    Nt_ex = 5000
    Nt = 10000
    Ntplot = 200
    tplot = range(0, Nt_ex, Nt_ex//Ntplot)

    print('Running sims for c={}'.format(c))
    # g = plot_example_and_hist(a=a, b=b, c=c, K=K, Nplot=Nplot, Ninit=Ninit, eta=eta, p=p, Nt_ex=Nt_ex, Nt=Nt)
    # mg0 = SeabornFig2Grid(g, fig, gs[0]) # copy it to the figure
    Jt = run_sim_return_traces(K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False)
    ax[0, 0].plot(Jt, linewidth=2)

    ax[0, 0].set_xticklabels([])
    ax[0, 0].set_xlabel('Time', fontsize=fontsize)

    bins = np.arange(.5*K**(-1/p), 1+.5*K**(-1/p), K**(-1/p))
    bins = np.sort(np.concatenate((-bins, bins, [0])))

    pars = [[2, 1], [1, 2]]

    for a, c in pars:

        Jfinal = final_weight_dist(K=K, Nt=Nt, a=a, b=b, c=c, p=p, flat_init=False, Ninit=Ninit, eta=eta)
        ax[1, 0].hist(Jfinal.reshape(-1,), bins=bins, alpha=0.5, label='(a,c)=({},{})'.format(a, c))

    ax[1, 0].set_ylabel('Count', fontsize=fontsize)
    ax[1, 0].set_xlabel('Synaptic weight', fontsize=fontsize)
    ax[1, 0].legend(loc=0, frameon=False, fontsize=fontsize)
    ax[1, 0].set_yscale('log')
    ax[0, 0].set_title('Odd a+c>1')
    ax[0, 0].set_ylabel('Synaptic weight', fontsize=fontsize)
    ax[1, 0].set_ylim((1, K*Ninit))
    ax[1, 0].set_yticks((1, Ninit/2, Ninit, K*Ninit))

    # # l0_norm = loop_initial_cond_lp(Ninit=Ninit, K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=True, thresh=thresh, p_calc=0)
    # # plot_std = np.nanstd(l0_norm, axis=1) / np.sqrt(Ninit)
    # # plot_mean = np.nanmean(l0_norm, axis=1)

    # # plot_std = plot_std[::Nt_ex//Ntplot]
    # # plot_mean = plot_mean[::Nt_ex//Ntplot]

    # # ax[1, 0].fill_between(tplot, plot_mean + plot_std, plot_mean - plot_std, alpha=0.2, color='k')
    # # ax[1, 0].plot(tplot, plot_mean, linewidth=2, color='k')

    # # ylim = ax[0, 0].get_ylim()
    # # ymax = max(np.abs(ylim))
    # # ax[0, 0].set_ylim((-ymax, ymax))
    # # ax[1, 0].set_ylim((.9, K))
    # # ax[1, 0].set_yticks(np.linspace(1, K, 4, dtype=int))


    # # ax[1, 0].set_ylabel('Nonzero synapses')

    a = 1
    c = 1

    print('Running sims for c={}'.format(c))
    Jt = run_sim_return_traces(K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False)

    ax[0, 1].plot(Jt, linewidth=2)

    pars = [[1, 1], [3, 1]]

    for a, c in pars:

        Jfinal = final_weight_dist(K=K, Nt=Nt, a=a, b=b, c=c, p=p, flat_init=False, Ninit=Ninit, eta=eta)
        counts, _ = np.histogram(Jfinal, bins=bins)
        ax[1, 1].hist(Jfinal.reshape(-1,), bins=bins, alpha=0.5, label='(a,c)=({},{})'.format(a, c))

    ax[1, 1].set_xlabel('Synaptic weight', fontsize=fontsize)
    ax[1, 1].legend(loc=0, frameon=False, fontsize=fontsize)
    ax[1, 1].set_yscale('log')
    ax[0, 1].set_title('Even a+c>1')
    ax[1, 1].set_ylim((1, K*Ninit))

    eta = .001
    a = 1
    c = -1
    # eta = .001
    # Nt_ex = 1000000
    # # Nt = 1000000
    # tplot = range(0, Nt_ex, Nt_ex//Ntplot)

    print('Running sims for c={}'.format(c))
    Jt = run_sim_return_traces(K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False)
    ax[0, 2].plot(Jt, linewidth=2)

    pars = [[1, -1], [2, -2]]

    for a, c in pars:

        Jfinal = final_weight_dist(K=K, Nt=Nt, a=a, b=b, c=c, p=p, flat_init=False, Ninit=Ninit, eta=eta)
        counts, _ = np.histogram(Jfinal, bins=bins)
        ax[1, 2].hist(Jfinal.reshape(-1,), bins=bins, alpha=0.5, label='(a,c)=({},{})'.format(a, c))

    ax[1, 2].set_xlabel('Synaptic weight', fontsize=fontsize)
    ax[1, 2].legend(loc=0, frameon=False, fontsize=fontsize)
    ax[1, 2].set_yscale('log')
    ax[0, 2].set_title('a+c=0')
    ax[1, 2].set_ylim((1, K*Ninit))

    a = 2
    c = -3

    print('Running sims for c={}'.format(c))
    Jt = run_sim_return_traces(K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False)
    ax[0, 3].plot(Jt, linewidth=2)
    ax[0, 3].set_title('Odd a+c<0')

    pars = [[1, -2], [2, -3]]

    for a, c in pars:

        Jfinal = final_weight_dist(K=K, Nt=Nt, a=a, b=b, c=c, p=p, flat_init=False, Ninit=Ninit, eta=eta)
        counts, _ = np.histogram(Jfinal, bins=bins)
        ax[1, 3].hist(Jfinal.reshape(-1,), bins=bins, alpha=0.5, label='(a,c)=({},{})'.format(a, c))

    ax[1, 3].set_xlabel('Synaptic weight', fontsize=fontsize)
    ax[1, 3].legend(loc=0, frameon=False, fontsize=fontsize)
    ax[1, 3].set_yscale('log')
    ax[1, 3].set_ylim((1, K*Ninit))

    # Nt_ex = 100000
    # tplot = range(0, Nt_ex, Nt_ex//Ntplot)

    # print('Running sims for c={}'.format(c))
    # # g = plot_example_and_hist(a=a, b=b, c=c, K=K, Nplot=Nplot, Ninit=Ninit, eta=eta, p=p, Nt_ex=Nt_ex, Nt=Nt)
    # # mg2 = SeabornFig2Grid(g, fig, gs[2]) # copy it to the figure

    # print('Running sims for c={}'.format(c))
    # need_sim = True

    # while need_sim:
    #     Jt = run_sim_return_traces(K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False)
    #     if ~np.any(~np.isfinite(Jt)):
    #         need_sim = False
    #     else:
    #         pass

    # ax[0, 3].plot(Jt, linewidth=2)

    # l0_norm = loop_initial_cond_lp(Ninit=Ninit, K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=True, thresh=thresh, p_calc=p)
    # print(l0_norm.shape)
    # plot_std = np.nanstd(l0_norm, axis=1) / np.sqrt(Ninit)
    # plot_mean = np.nanmean(l0_norm, axis=1)
    # plot_std = plot_std[::Nt_ex//Ntplot]
    # plot_mean = plot_mean[::Nt_ex//Ntplot]

    # ax[1, 3].fill_between(tplot, plot_mean + plot_std, plot_mean - plot_std, alpha=0.2, color='k')
    # ax[1, 3].plot(tplot, plot_mean, linewidth=2, color='k')
    # ax[1, 3].plot(tplot, np.ones(Ntplot), 'k--')

    # ax[0, 3].set_xticklabels([])
    # # ax[0, 3].set_ylim((-1.1, 1.1))
    # # ax[1, 3].set_ylim((1, K))
    # # ax[1, 3].set_yticks(np.linspace(1, K+1, 4, dtype=int))
    # ax[0, 3].set_title('(a,b,c)=({},{},{})'.format(a,b,c), fontsize=fontsize)
    # ax[1, 3].set_xlabel('Time')

    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(savefile)

    return None


def plot_fig_5(savefile='fig5.pdf', camp='cividis', K=10, Ninit=20, Nplot=None):
    
    '''
    not used
    plot figure for convergence to Lp sphere with a+c=1
    '''

    if (Nplot is None) or (Nplot > K):
        Nplot = K

    fig, ax = plt.subplots(2, 2, figsize=(3.5, 3.5))

    a = 1
    b = 1
    c = 0
    p = 2
    eta = .01
    Nplot = K

    Nt_ex=3000
    Ntplot = 200
    tplot = range(0, Nt_ex, Nt_ex//Ntplot)

    print('Running sims for c={}'.format(c))
    Jt = run_sim_return_traces(K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False)
    
    ax[0, 0].plot(Jt, linewidth=2)

    p_plot = [1, 2]
    norm_init_factors = [.5, 1, 2]

    for i, p in enumerate(p_plot):
        for j, norm_init in enumerate(norm_init_factors):

            lp_norm = loop_initial_cond_lp(Ninit=Ninit, K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False, p_calc=p, norm_init_factor=norm_init)
            plot_std = np.nanstd(lp_norm, axis=1) / np.sqrt(Ninit)
            plot_mean = np.nanmean(lp_norm, axis=1)

            plot_std = plot_std[::Nt_ex//Ntplot]
            plot_mean = plot_mean[::Nt_ex//Ntplot]

            ax[1, 0].fill_between(tplot, plot_mean + plot_std, plot_mean - plot_std, alpha=0.2, color=colors[i])
            if j == 0:
                if i == 0:
                    ax[1, 0].plot(tplot, plot_mean, linewidth=2, color=colors[i], label='p={}'.format(p))
                else:
                    ax[1, 0].plot(tplot, plot_mean, '--', linewidth=2, color=colors[i], label='p={}'.format(p))
            else:
                if i == 0:
                    ax[1, 0].plot(tplot, plot_mean, linewidth=2, color=colors[i])
                else:
                    ax[1, 0].plot(tplot, plot_mean, '--', linewidth=2, color=colors[i])


    ax[1, 0].legend(loc=0, frameon=False, fontsize=fontsize)
    
    ax[0, 0].set_xticklabels([])
    ax[0, 0].set_ylim((-1.1, 1.1))
    # ax[1, 0].set_ylim((0, 2))
    # ax[1, 0].set_yticks(np.linspace(1, K, 4, dtype=int))
    ax[0, 0].set_title('(a,b,c)=({},{},{})'.format(a,b,c), fontsize=fontsize)
    ax[1, 0].set_xlabel('Time')

    
    a = 2
    b = 1
    c = -1

    print('Running sims for c={}'.format(c))
    need_sim = True

    while need_sim:
        Jt = run_sim_return_traces(K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False)
        if ~np.any(~np.isfinite(Jt)):
            need_sim = False
        else:
            pass

    ax[0, 1].plot(Jt, linewidth=2)

    p_plot = [1, 2]
    norm_init_factors = [.5, 1, 2]

    for i, p in enumerate(p_plot):
        for j, norm_init in enumerate(norm_init_factors):

            lp_norm = loop_initial_cond_lp(Ninit=Ninit, K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False, p_calc=p, norm_init_factor=norm_init)
            plot_std = np.nanstd(lp_norm, axis=1) / np.sqrt(Ninit)
            plot_mean = np.nanmean(lp_norm, axis=1)

            plot_std = plot_std[::Nt_ex//Ntplot]
            plot_mean = plot_mean[::Nt_ex//Ntplot]

            ax[1, 1].fill_between(tplot, plot_mean + plot_std, plot_mean - plot_std, alpha=0.2, color=colors[i])
            if j == 0:
                if i == 0:
                    ax[1, 1].plot(tplot, plot_mean, linewidth=2, color=colors[i], label='p={}'.format(p))
                else:
                    ax[1, 1].plot(tplot, plot_mean, '--', linewidth=2, color=colors[i], label='p={}'.format(p))
            else:
                if i == 0:
                    ax[1, 1].plot(tplot, plot_mean, linewidth=2, color=colors[i])
                else:
                    ax[1, 1].plot(tplot, plot_mean, '--', linewidth=2, color=colors[i])

    ax[1, 1].legend(loc=0, frameon=False, fontsize=fontsize)
    
    ax[0, 1].set_xticklabels([])
    ax[0, 1].set_ylim((-1.1, 1.1))
    # ax[1, 0].set_ylim((0, 2))
    # ax[1, 0].set_yticks(np.linspace(1, K, 4, dtype=int))
    ax[0, 1].set_title('(a,b,c)=({},{},{})'.format(a,b,c), fontsize=fontsize)
    ax[1, 1].set_xlabel('Time')

    # ax[1, 0].set_ylim((0.75, 2.25))
    # ax[1, 1].set_ylim((0.75, 2.25))

    ax[0, 0].set_ylabel('Synaptic weight', fontsize=fontsize)
    ax[1, 0].set_ylabel('Synaptic weight norm', fontsize=fontsize)

    # ax[0, 1].set_ylabel('Synaptic weight', fontsize=fontsize)
    # ax[1, 1].set_ylabel('Nonzero synapses', fontsize=fontsize)

    # ymax = np.round(ax[1, 1].get_ylim()[1])
    # ax[1, 1].set_yticks(np.linspace(1, ymax, 4, dtype=int))
    # ax[1, 1].set_yticks(range(1, 9, 2))

    sns.despine(fig)
    fig.tight_layout()
    # gs.tight_layout(fig)

    fig.savefig(savefile)


def plot_fig_ac_even_negative(savefile='fig6.pdf', cmap='cividis', K=10, Ninit=20, Nplot=None):
    
    '''
    '''

    fig, ax = plt.subplots(1, 4, figsize=(7, 1.75))

    a = 1
    b = 1
    c = -3
    p = 2
    eta = .001
    Nt = 5000

    Jt = run_sim_return_traces(a=a, b=b, c=c, p=p, eta=eta, Nt=Nt)

    T = Jt.shape[0]
    Ntplot = 400
    tplot = range(0, T, T//Ntplot)

    ax[0].plot(Jt, linewidth=2)
    ax[0].set_title('(a,c)=({},{})'.format(a, c), fontsize=fontsize)

    pnorm = loop_initial_cond_lp(a=a, b=b, c=c, p=p, p_calc=p, eta=eta, Ninit=Ninit, Nt=40000)
    T = pnorm.shape[0]
    tplot = range(0, T, T//Ntplot)

    norm_mean = np.nanmean(pnorm, axis=1)
    norm_std = np.nanstd(pnorm, axis=1) / np.sqrt(Ninit)
    norm_mean = norm_mean[tplot]
    norm_std = norm_std[tplot]

    ax[3].fill_between(tplot, norm_mean + norm_std, norm_mean - norm_std, alpha=0.2, color=colors[0])
    ax[3].plot(tplot, norm_mean, color=colors[0], linewidth=2, label='(a,c)=({},{})'.format(a, c))


    a = 2
    c = -4
    Nt = 40000

    Jt = run_sim_return_traces(a=a, b=b, c=c, p=p, eta=eta, Nt=Nt)
    T = Jt.shape[0]
    tplot = range(0, T, T//Ntplot)

    ax[1].plot(Jt, linewidth=2)
    ax[1].set_title('(a,c)=({},{})'.format(a, c), fontsize=fontsize)

    pnorm = loop_initial_cond_lp(a=a, b=b, c=c, p=p, p_calc=p, eta=eta, Ninit=Ninit, Nt=40000)
    T = pnorm.shape[0]
    tplot = range(0, T, T//Ntplot)

    norm_mean = np.nanmean(pnorm, axis=1)
    norm_std = np.nanstd(pnorm, axis=1) / np.sqrt(Ninit)
    norm_mean = norm_mean[tplot]
    norm_std = norm_std[tplot]

    ax[3].fill_between(tplot, norm_mean + norm_std, norm_mean - norm_std, alpha=0.2, color=colors[1])
    ax[3].plot(tplot, norm_mean, color=colors[1], linewidth=2, label='(a,c)=({},{})'.format(a, c))

    
    a = 1
    c = -5
    Nt = 5000

    Jt = run_sim_return_traces(a=a, b=b, c=c, p=p, eta=eta, Nt=Nt)
    T = Jt.shape[0]
    tplot = range(0, T, T//Ntplot)

    ax[2].plot(Jt, linewidth=2)
    ax[2].set_title('(a,c)=({},{})'.format(a, c), fontsize=fontsize)

    pnorm = loop_initial_cond_lp(a=a, b=b, c=c, p=p, p_calc=p, eta=eta, Ninit=Ninit, Nt=40000)
    T = pnorm.shape[0]
    tplot = range(0, T, T//Ntplot)

    norm_mean = np.nanmean(pnorm, axis=1)
    norm_std = np.nanstd(pnorm, axis=1) / np.sqrt(Ninit)
    norm_mean = norm_mean[tplot]
    norm_std = norm_std[tplot]

    ax[3].fill_between(tplot, norm_mean + norm_std, norm_mean - norm_std, alpha=0.2, color=colors[2])
    ax[3].plot(tplot, norm_mean, color=colors[2], linewidth=2, label='(a,c)=({},{})'.format(a, c))

    ax[0].set_ylabel('Synaptic weight', fontsize=fontsize)
    ax[3].set_ylabel('Synaptic weight norm', fontsize=fontsize)
    ax[3].legend(loc=0, frameon=False, fontsize=fontsize)


    yaxlim = np.amax(np.abs([ax[0].get_ylim(), ax[1].get_ylim(), ax[2].get_ylim()]))
    
    for i in range(1, 3):
        ax[i].set_yscale('symlog')
        ax[i].set_xlabel('Time', fontsize=fontsize)

    if np.amax(np.abs(ax[0].get_ylim())) < 1:
        ax[0].set_ylim((-1, 1))
        ax[0].set_yticks((-1, 0, 1))

    ax[1].set_yticks((-100, -10, 0, 10, 100))
    ax[2].set_yticks((-1000, -10, 0, 10, 1000))

    ax[3].set_xlabel('Time', fontsize=fontsize)
    ax[3].set_yscale('log')
    ax[3].set_yticks((1, 1000, 1000000))

    sns.despine(fig)
    fig.tight_layout()
    fig.savefig(savefile)

    return None

def plot_fig_5_old(savefile='fig5.pdf', cmap='cividis', K=10, Ninit=20, Nplot=10):

    '''
    a+c odd and not =1
    '''

    fig, ax = plt.subplots(2, 2, figsize=(3.5, 3.5))

    a = 2
    b = 1
    c = -3
    p = 2
    eta = .01
    Nplot = K

    Nt_ex=20000
    Ntplot = 200
    tplot = range(0, Nt_ex, Nt_ex//Ntplot)

    print('Running sims for c={}'.format(c))
    Jt = run_sim_return_traces(K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False)
    
    ax[0, 0].plot(Jt, linewidth=2)

    p_plot = [1, 2]
    norm_init_factors = [1]

    for i, p in enumerate(p_plot):
        for j, norm_init in enumerate(norm_init_factors):

            lp_norm = loop_initial_cond_lp(Ninit=Ninit, K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False, p_calc=p, norm_init_factor=norm_init)
            plot_std = np.nanstd(lp_norm, axis=1) / np.sqrt(Ninit)
            plot_mean = np.nanmean(lp_norm, axis=1)

            plot_std = plot_std[::Nt_ex//Ntplot]
            plot_mean = plot_mean[::Nt_ex//Ntplot]

            ax[1, 0].fill_between(tplot, plot_mean + plot_std, plot_mean - plot_std, alpha=0.2, color=colors[i])
            if j == 0:
                if i == 0:
                    ax[1, 0].plot(tplot, plot_mean, linewidth=2, color=colors[i], label='p={}'.format(p))
                else:
                    ax[1, 0].plot(tplot, plot_mean, '--', linewidth=2, color=colors[i], label='p={}'.format(p))
            else:
                if i == 0:
                    ax[1, 0].plot(tplot, plot_mean, linewidth=2, color=colors[i])
                else:
                    ax[1, 0].plot(tplot, plot_mean, '--', linewidth=2, color=colors[i])


    ax[1, 0].legend(loc=0, frameon=False, fontsize=fontsize)
    
    ax[0, 0].set_xticklabels([])
    # ax[0, 0].set_ylim((-1.1, 1.1))
    # ax[1, 0].set_ylim((0, 2))
    # ax[1, 0].set_yticks(np.linspace(1, K, 4, dtype=int))
    ax[1, 0].set_yscale('log')
    ax[0, 0].set_title('(a,b,c)=({},{},{})'.format(a,b,c), fontsize=fontsize)
    ax[1, 0].set_xlabel('Time')


    a = 2
    c = -5
    p = 4
    eta = .01
    Nplot = K

    Nt_ex=10000
    tplot = range(0, Nt_ex, Nt_ex//Ntplot)

    print('Running sims for c={}'.format(c))
    Jt = run_sim_return_traces(K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False)
    
    ax[0, 1].plot(Jt, linewidth=2)

    p_plot = [1, 2]
    norm_init_factors = [1]

    for i, p in enumerate(p_plot):
        for j, norm_init in enumerate(norm_init_factors):

            lp_norm = loop_initial_cond_lp(Ninit=Ninit, K=K, Nt=Nt_ex, a=a, b=b, c=c, p=p, eta=eta, flat_init=False, p_calc=p, norm_init_factor=norm_init)
            plot_std = np.nanstd(lp_norm, axis=1) / np.sqrt(Ninit)
            plot_mean = np.nanmean(lp_norm, axis=1)

            plot_std = plot_std[::Nt_ex//Ntplot]
            plot_mean = plot_mean[::Nt_ex//Ntplot]

            ax[1, 1].fill_between(tplot, plot_mean + plot_std, plot_mean - plot_std, alpha=0.2, color=colors[i])
            if j == 0:
                ax[1, 1].plot(tplot, plot_mean, linewidth=2, color=colors[i], label='p={}'.format(p))
            else:
                ax[1, 1].plot(tplot, plot_mean, linewidth=2, color=colors[i])

    ax[1, 1].legend(loc=0, frameon=False, fontsize=fontsize)
    
    ax[0, 1].set_xticklabels([])
    # ax[0, 0].set_ylim((-1.1, 1.1))
    # ax[1, 0].set_ylim((0, 2))
    # ax[1, 0].set_yticks(np.linspace(1, K, 4, dtype=int))
    ax[1, 1].set_yscale('log')
    ax[0, 1].set_title('(a,b,c)=({},{},{})'.format(a,b,c), fontsize=fontsize)
    ax[1, 1].set_xlabel('Time')

    ax[0, 0].set_yscale('symlog')
    ax[0, 1].set_yscale('symlog')
    ax[0, 0].set_ylabel('Synaptic weight', fontsize=fontsize)
    ax[1, 0].set_ylabel('Synaptic weight norm', fontsize=fontsize)


    sns.despine(fig)
    fig.tight_layout()
    # gs.tight_layout(fig)

    fig.savefig(savefile)

if __name__ == '__main__':
    plot_fig_ac_even_negative()
    # plot_fig_3(Ninit=20, K=10)
    # plot_fig_4(Ninit=50, K=10)
    # plot_fig_5(Ninit=20, K=10)