# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 09:04:07 2015

@author: gabeo

define a class to hold parameters
"""

import numpy as np

class params:
    def __init__(self):
        # self.Ne = 100
        # self.Ni = 50
        # self.N = self.Ne+self.Ni
        # self.pEE = .5
        # self.pEI = 1.
        # self.pIE = 1.
        # self.pII = 1.

        self.Ne = 1
        self.Ni = 0
        self.N = self.Ne+self.Ni
        self.pEE = 1.
        self.pEI = 1.
        self.pIE = 1.
        self.pII = 1.

        # self.WmaxE = 0.
        # self.weightEE = self.WmaxE
        # self.weightIE = 0.
        # self.weightEI = 0.
        # self.weightII = 0.

        if self.Ne > 0:
            # self.WmaxE = 1.99/(self.Ne*self.pEE)
            self.WmaxE = 1. - 1e-6
            self.weightEE = .75/(self.Ne*self.pEE)

            # self.WmaxE = 1.5
            # self.weightEE = 1. / (self.Ne*self.pEE)

            self.weightIE = .5 / (self.Ne*self.pIE)
        else:
            self.WmaxE = 0.
            self.weightEE = 0.
            self.weightIE = 0.

        if self.Ni > 0:
            self.weightEI = -1.1 / (self.Ni*self.pEI) # -1
            self.weightII = -1. / (self.Ni*self.pII)
        else:
            self.weightEI = 0.
            self.weightII = 0.

        # self.tau = 10.*np.ones(self.N, dtype=np.float32) # synaptic time constant
        self.tau = 10.
        self.tauE = 15.
        self.tauI = 10.
        self.taud = 0

        self.gain = 10. # coefficient for threshold power-law input-output function
        self.p = 1. # power for threshold power-law input-output function
        self.b = .05 * np.ones(self.N, dtype=np.float32) # baseline rate, sp/ms, .015
        # self.b *= 0.015

        ### for e-i network
        # self.b[:self.Ne] *= .015
        # self.b[self.Ne:] *= .01
        ## self.b *= .015

        self.A2plus = 15e-3
        # self.A3plus = 1.3e-2
        self.A2minus = 7.1e-3
        self.A3plus = 0
        self.A3minus = 0
        self.tauplus = 17 # msec
        self.tauminus = 34 # msec
        self.taux = 101 # msec
        self.tauy = 114 # msec
        self.p_plast = 2

        self.r_thresh = 20./1000 # BCM potentiation threshold
        # self.r_thresh = - (-self.A2minus * self.tauminus + self.A3plus * self.tauplus) / (self.A3plus * self.tauplus * self.tauy)
        self.multiplicative_power = 0.

        # self.tau_bcm = 1*1e3*60 # time constant for post-synaptic rate-dependent averaging

        # self.eta = 1*1e-2 # learning rate parameter
        # self.eta = 0.
        self.eta = .01
        # self.eta = 1./(self.Ne)

        # self.tau_bcm = 150 / self.eta # keep >= ~5000 * Ne, otherwise faster than plasticity not same timescale
        self.tau_bcm = 5000.
