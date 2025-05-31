#!/usr/bin/env python

import strapdown
import numpy as np
import create_F
def prediction(dt ,P,Q, phi , la , h, v_n,v_e,v_d,R , f_ib_b , omega_ib_b,b_a,b_g):
    #print("barak")
    f_N,f_E ,f_D = f_ib_b
    p_new , v_new , R_new =  strapdown.strapdown(dt , phi , la , h, v_n,v_e,v_d,R , f_ib_b , omega_ib_b,b_a,b_g)
    P_new = create_F.propagate_covariance(P,dt,Q,R,phi,v_n,v_e,v_d,f_N,f_E ,f_D)
    phi , la , h = p_new
    v_n , v_e , v_d = v_new
    return  phi , la , h ,  v_n , v_e , v_d , R , P_new

def test_prediction():
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
    Q = np.arange(144).reshape(12,12)
    P = np.arange(15*15).reshape(15,15)
    phi , la , h ,  v_n , v_e , v_d , R , P_new = prediction(dt ,P,Q, phi , la , h, v_n,v_e,v_d,R , f_ib_b , omega_ib_b,b_a,b_g)
    return  phi , la , h ,  v_n , v_e , v_d , R , P_new


#phi  , *rest = test_prediction()

