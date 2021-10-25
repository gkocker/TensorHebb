import numpy as np
import os, pickle
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
# import SeabornFig2Grid as sfg
from scipy.linalg import norm

fontsize = 10
labelsize = 8

def plot_phase_portrait(ax, a=2, r=1, p=2):

    Nmesh = 6
    L = np.linspace(0, 2, Nmesh)
    M = np.linspace(-3, 3, Nmesh)

    LL, MM = np.meshgrid(L, M)

    Ldot = p * r**(a+1) * MM**a * LL * (1.-LL)
    Mdot = r**(a+1) * MM**(a+1) * (1.-LL)

    Ldot /= np.sqrt(Ldot**2 + Mdot**2)
    Mdot /= np.sqrt(Ldot**2 + Mdot**2)

    ax.quiver(LL, MM, Ldot, Mdot, scale=13, width=0.01)#, scale=10, width=0.01, headwidth=4, headlength=4)
    
    ax.plot(np.linspace(min(L)-.1*max(np.abs(L)), max(L)+.1*max(np.abs(L)), Nmesh), np.zeros(Nmesh,), 'k', linewidth=2)
    ax.plot(np.ones(Nmesh), np.linspace(min(M)-.1*max(np.abs(M)), max(M)+.1*max(np.abs(M)), Nmesh), 'k', linewidth=2)

    ax.set_yticks((min(M), 0, max(M)))
    ax.set_xticks((0, 1, 2))
    ax.set_xlabel('L', fontsize=fontsize)
    ax.set_ylabel('M', fontsize=fontsize)
    ax.set_title('(a, r)=({}, {})'.format(a, r), fontsize=fontsize)

    return ax

def plot_rank_one_LM(savefile='fig6.pdf'):

    fig, ax = plt.subplots(2, 3, figsize=(4.75, 3.75))

    ax1 = plot_phase_portrait(ax[0, 0], a=1, r=1)
    ax2 = plot_phase_portrait(ax[1, 0], a=3, r=1)

    ax3 = plot_phase_portrait(ax[0, 1], a=2, r=1)
    ax4 = plot_phase_portrait(ax[1, 1], a=4, r=1)

    ax5 = plot_phase_portrait(ax[0, 2], a=2, r=-1)
    ax6 = plot_phase_portrait(ax[1, 2], a=4, r=-1)

    for axi in [ax3, ax4, ax5, ax6]:
        axi.set_ylabel(None)
        axi.set_yticklabels([])

    # for axi in [ax1, ax3, ax5]:
    #     axi.set_xlabel(None)

    fig.tight_layout()
    fig.savefig(savefile)

    return None


def Ndot(p, r, a, M, L):

    N = np.arange(-3, 3)

    return N, p * r**(a+1) * M**a * (1-L) * N

def plot_rank_one_N(savefile='fig7.pdf', p=2, r=1):

    ### N vs Ndot for the 4 quadrants of L, M

    fig, ax = plt.subplots(2, 2, figsize=(3.75, 3.75))

    N, dN = Ndot(p=p, r=r, a=1, L=0.5, M=-1)
    ax[0, 0].plot(N, dN, linewidth=2)
    ax[0, 0].plot(N, 0*N, 'k')

    N, dN = Ndot(p=p, r=r, a=1, L=0.5, M=1)
    ax[0, 1].plot(N, dN, linewidth=2)
    ax[0, 1].plot(N, 0*N, 'k')

    N, dN = Ndot(p=p, r=r, a=1, L=1.5, M=-1)
    ax[1, 0].plot(N, dN, linewidth=2)
    ax[1, 0].plot(N, 0*N, 'k')

    N, dN = Ndot(p=p, r=r, a=1, L=1.5, M=1)
    ax[1, 1].plot(N, dN, linewidth=2)
    ax[1, 1].plot(N, 0*N, 'k')

    for axi in ax.reshape(-1,):
        axi.set_ylabel(r'$\dot{N}$', fontsize=fontsize)
        axi.set_xlabel('N', fontsize=fontsize)

    fig.tight_layout()
    fig.savefig(savefile)

if __name__ == '__main__':
    plot_rank_one_LM()
    # plot_rank_one_N()