#!/usr/bin/env python
import numpy as np
from scipy.linalg import expm

#np.set_printoptions(precision=0)
def create_F():
    omega_ie = 7.29 *10**(-5)
    g = 9.81
    R_e  = 6.371 * 10**(6)
    Omega_N =lambda phi : omega_ie*np.cos(phi)
    Omega_D = lambda phi : omega_ie* np.sin(phi)
    rho_N = lambda v_e : v_e / R_e
    rho_E = lambda v_n :  -v_n /R_e
    rho_D = lambda v_e ,phi:  v_e *np.tan(phi) /R_e
    omega_N = lambda phi , v_e  : Omega_N(phi) + rho_N(v_e)
    omega_E = lambda v_n : rho_E(v_n)
    omega_D = lambda phi , v_e: Omega_D(phi) + rho_D(v_e,phi)
    k_D = lambda v_d : v_d / R_e
    F41 = lambda  phi , v_e: -2*Omega_N(phi) * v_e - rho_N(v_e)*v_e /np.cos(phi)**2
    F43 = lambda phi, v_n,v_e ,v_d : rho_E(v_n) *k_D(v_d) - rho_N(v_e)* rho_D(v_e,phi)
    F51 = lambda phi , v_n ,v_e,v_d:2*(Omega_N(phi)*v_n + Omega_D(phi)*v_d) + rho_N(v_e)*v_n/np.cos(phi)**2
    F53 = lambda phi,v_n,v_e,v_d : rho_E(v_d)*rho_D(v_e , phi)-k_D(v_d)*rho_N(v_e)
    F54 = lambda phi , v_e: -( omega_D(phi,v_e) +Omega_D(phi))
    F55 = lambda phi , v_n ,v_d: k_D(v_d) - rho_E(v_n)*np.tan(phi)
    F56 = lambda  phi , v_e: omega_N(phi,v_e) + Omega_N(phi)
    F63 = lambda  v_n ,v_e: rho_N(v_e)**2 + rho_E(v_n)**2 - 2*g/R_e
    F91 = lambda  phi , v_e: Omega_N(phi)+ rho_N(v_e)/np.cos(phi)**2

def create_F_9_by_9(phi,v_n,v_e,v_d,f_N,f_E ,f_D):
    omega_ie = 7.29 *10**(-5)
    g = 9.81
    R_e  = 6.371 * 10**(6)
    Omega_N = omega_ie*np.cos(phi)
    Omega_D =  omega_ie* np.sin(phi)
    rho_N =  v_e / R_e
    rho_E =   -v_n /R_e
    rho_D =  v_e *np.tan(phi) /R_e
    omega_N =  Omega_N + rho_N
    omega_E =  rho_E
    omega_D =  Omega_D + rho_D
    k_D =  v_d / R_e
    F41 =  -2*Omega_N * v_e - rho_N*v_e /np.cos(phi)**2
    F43 = rho_E *k_D - rho_N* rho_D
    F51 = 2*(Omega_N*v_n + Omega_D*v_d) + rho_N*v_n/np.cos(phi)**2
    F53 =  rho_E*rho_D-k_D*rho_N
    F54 =  -( omega_D +Omega_D)
    F55 =  k_D - rho_E*np.tan(phi)
    F56 =  omega_N + Omega_N
    F63 = rho_N**2 + rho_E**2 - 2*g/R_e
    F91 =  Omega_N+ rho_N/np.cos(phi)**2
    F = np.zeros((9,9))
    F[1,0] = -rho_D/np.cos(phi)
    F[0,2] = rho_E/R_e
    F[1,2]= -rho_N / (R_e *np.cos(phi))

    F[0,3]= 1/R_e
    F[1,4]= 1/(R_e*np.cos(phi))
    F[2,5]= -1

    A = np.zeros((3,3))
    A[0,0] = F41
    A[1,0]= F51
    A[2,0]= -2*v_e*Omega_D
    A[0,2] = F43
    A[1,2]= F53
    A[2,2] = F63
    F[3:6,0:3] = A


    # F 22 
    A = np.zeros((3,3))
    A[0,0] = k_D
    A[1,0]= F54
    A[2,0] = 2*rho_E
    A[0,1] = 2*omega_D
    A[1,1] = F55
    A[1,2] = -2*omega_N
    A[0,2] = -rho_E
    A[1,2] = F56
    F[3:6,3:6] = A

    # F23
    A = np.zeros((3,3))
    A[1,0]= -f_D
    A[2,0] = f_E
    A[0,1] = f_D
    A[2 , 1] = -f_N
    A[0,2] = -f_E
    A[1,2] = f_N
    F[3:6,6:9] = A

    # F31
    A = np.zeros((3,3))
    A[0,0]= -Omega_D
    A[0,2]= rho_N/R_e
    A[1,2] = rho_E / R_e
    A[2,0]= F91
    A[2,2]= rho_D/R_e
    F[6:9,0:3] = A

    # F32
    A = np.zeros((3,3))
    A[0,1]= -1/R_e
    A[1,0]= 1/R_e
    A[2 , 1] = np.tan(phi)/R_e
    F[6:9,3:6] = A

    # F33
    A = np.zeros((3,3))
    A[0,1]= omega_D
    A[0,2]= -omega_E
    A[1,0]= -omega_D
    A[1,2] = omega_N
    A[2,0]= omega_E
    A[2,1]= -omega_N
    F[6:9,6:9] = A

    #print(F)

    return F 

def create_F_15_by_15(phi,v_n,v_e,v_d , f_N,f_E,f_D,R):
    F = np.zeros((15,15))
    F[0:9,0:9] = create_F_9_by_9(  phi,v_n,v_e,v_d , f_N,f_E,f_D  )
    #F[9:12 , 9:12] = -0.001 *np.identity(3)
    #F[12:15 , 12:15] = -0.0001 * np.identity(3)
    F[9:12 , 9:12] = 0.1 *np.identity(3)
    F[12:15 , 12:15] = 0.1 * np.identity(3)
    F[3:6 , 9:12] = -R # np.identity(3) # should be -R
    F[6:9 , 12:15] = R #np.identity(3) # should be R
    return F
def create_G(R_b_n):
    """
    noise_vector = 
    noise acc 
    noise gyro
    noise bias_acc
    noise bias_gyro

    R_b_n - b is subscript , n is superscript
    """
    G = np.zeros((15,12))
    G[3:6,0:3] = -R_b_n
    G[6:9,3:6] = -R_b_n
    G[9:12,6:9] = np.identity(3)
    G[12:15,9:12] = np.identity(3)
    #print(G)
    return G 

def propagate_covariance(P_old,dt,Q,R_b_n,phi,v_n,v_e,v_d,f_N,f_E ,f_D):
    F = create_F_15_by_15(phi,v_n,v_e,v_d , f_N,f_E,f_D,R_b_n)
    #print("Barak")
    G = create_G(R_b_n)
    Phi_prop= np.identity(15) + dt* F 
    return Phi_prop@ P_old@ Phi_prop.T + G@ Q @G.T *dt     

def test_create_F():
    create_F_9_by_9(np.pi/2,2,3,4,6,6,6)
    R = np.arange(9).reshape(3,3)
    create_F_15_by_15(np.pi/2,2,3,4,6,6,6,R)
def test_propagate_cov():
    R_n_b = np.arange(9).reshape(3,3)
    Q = np.arange(144).reshape(12,12)
    P = np.arange(15*15).reshape(15,15)
    propagate_covariance( P ,0.01,Q,R_n_b, np.pi/2,2,3,4,6,6,6)
def test_create_G():
    R_n_b = np.arange(9).reshape(3,3)
    create_G(R_n_b)
    create_F()
