# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:08:02 2015

@author: gabeo

define activation function for Poisson model
right now, is the half-wave rectified input
"""


''' rectified linear '''
def phi(g,gain):
    
    import numpy as np
    ind = np.where(g<0)
    g[ind[0]] = 0

    r_out = gain*g

    return(r_out)

def phi_prime(g,gain):

    import numpy as np
    ind = np.where(g<0)

    phi_pr = gain*np.ones(np.shape(g))
    phi_pr[ind] = 0

    return phi_pr


# ''' rectified quadratic '''
# def phi(g, gain):
#     import numpy as np
#     ind = np.where(g < 0)
#     g[ind[0]] = 0.
#
#     r_out = gain * g**2
#
#     return (r_out)
#
#
# def phi_prime(g, gain):
#     import numpy as np
#     ind = np.where(g < 0)
#
#     phi_pr = 2 * gain * g
#     phi_pr[ind] = 0
#
#     return phi_pr
