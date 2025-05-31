#!/usr/bin/env python

R_E = 6378137.0  # Earth's equatorial radius [m]
omega_ie = 7.292115e-5  # Earth's rotation rate [rad/s]
g_0 = 9.81  # Gravity magnitude [m/s^2]
R_m =  6378137.0 # To change 
R_n =  6378137.0# To change 

def skew_symmetric(v):
    """Returns the skew-symmetric matrix of a vector v."""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
import numpy as np
def strapdown(dt , phi , la , h, v_n,v_e,v_d,R , f_ib_b , omega_ib_b,bias_f,bias_g):

    new_phi = phi + dt *( v_n /(R_m+h) )
    new_la = la + dt * v_e / (np.cos(phi)* (R_n + h))
    new_h = h + dt * (-v_d)

    p_new = np.array([new_phi , new_la, new_h])
    v = np.array([v_n , v_e , v_d])
    g_n = np.array([0,0,g_0])
    omega_en_n =  np.array([v_e / (np.cos(phi)* (R_n + h)) * np.cos(phi),
                      ( v_n /(R_m+h) )  ,
                      - v_e / (np.cos(phi)* (R_n + h)) *np.sin(phi)] )
    Omega_en_n = skew_symmetric(omega_en_n)
    omega_ie_n = np.array([ omega_ie*np.cos(phi),
                   0 ,
                   -omega_ie*np.sin(phi)])
    Omega_ie_n = skew_symmetric(omega_ie_n)
    v_new = v + dt*(R @ (f_ib_b - bias_f ) + g_n - (Omega_en_n + 2*Omega_ie_n) @ v)
    #print(R @ (f_ib_b - bias_f ) + g_n)

    omega_in_n = omega_en_n + omega_ie_n
    omega_in_b = R @ omega_in_n
    Omega_in_b = skew_symmetric(omega_in_b)
    R_new = R + R@ ( skew_symmetric(omega_ib_b - bias_g) -  Omega_in_b) * dt

    return p_new , v_new , R_new
    

def test_strapdown():
    R = np.arange(9).reshape(3,3)
    p = np.arange(3)
    v = np.arange(3)
    b_a = np.arange(3)
    b_g = np.arange(3)
    f_ib_b = np.arange(3)
    omega_ib_b = np.arange(3)
    dt = 0.01 
    phi , la ,h = p 
    v_n , v_e , v_d = v
    strapdown(dt , phi , la , h, v_n,v_e,v_d,R , f_ib_b , omega_ib_b,b_a,b_g)

#test_strapdown()
