#!/usr/bin/env python
import simulation_interface
import strapdown
import numpy as np

Q = np.eye(12)
Q[0, 0] *= 20.00**2 # ACC 
Q[1, 1] *= 20.00**2
Q[2, 2] *= 0.004**2
Q[3, 3] *= 0.00004**2 # GYRO 
Q[4, 4] *= 0.00004**2
Q[5, 5] *= 0.04**2
Q[6:9, 6:9] *= 1e-9
Q[9:12, 9:12] *= 1e-9
radius = 50000
omega = 2 * np.pi / 60
#func_acc_gt = lambda t : [-radius * omega**2 * np.cos(omega * t) ,  -radius * omega**2 * np.sin(omega * t) ,0]
#func_acc_gt = lambda t : [omega  , omega + np.sin(t)  ,0]
#func_gyro_gt = lambda t : [ 0.1 ,  -0.1 ,0]

#func_acc_gt = lambda t : [0  , 0  ,9.81]
func_acc_gt = lambda t : [0  , 0  ,0]
func_gyro_gt = lambda t : [ 0 ,  0 ,0]

err_phi = 0.0001
err_la = 0.0001
err_la = 0.0001
p_gt = np.array([np.pi /4 , np.pi/4 , 0])
v_gt = np.array([ 14,41,4])
R_gt = np.identity(3)
b_a_gt = np.array([0.5, 0.2, 0.5])
b_g_gt = np.array([0.01, -0.02, 0.005])
x_gt = (p_gt,v_gt , R_gt , b_a_gt , b_g_gt)

p_est = np.array([np.pi/4 + 0.0001 , np.pi/4 + 0.0001, 0])
v_est = np.array([ 14,41,4])
R_S = strapdown.skew_symmetric([0.2,0.4,-0.3])
R_est = np.identity(3)
#R_est = alg.expm(R_S)
b_a_est = np.array([4.5, 10.2, -10.5])
#b_a_est = np.array([40.5, -100.2, -1000.5])
b_g_est = np.array([0.01, -0.02, 0.005])
#b_a_est = np.array([10.5, -10.2, -2.5])
#b_g_est = np.array([1.0, -0.20, 0.5])
x_est = (p_est,v_est , R_est , b_a_est , b_g_est)
P = np.eye(15) * 1.0
P[0,0] *= 0.000000003
P[1,1] *= 0.000000003
P[3:6, 3:6] *= 100
P[9:12, 9:12] *= 10**2
P[12:15, 12:15] *= 1

R_gps = np.eye(3) * 0.00009  # 3 m std dev
R_gps[0,0] =0.0000000001  # 3 m std dev
R_gps[1,1] =0.0000000001  # 3 m std dev
R_gps[2,2] = 0.0000009  # 3 m std dev
#simulation(30,func_acc_gt , func_gyro_gt , Q , x_gt ,x_est, P)
#accel_noise_std = np.array([np.sqrt(Q[0, 0]) , np.sqrt(Q[1, 1]) ,np.sqrt(Q[2, 2]) ])
#gyro_noise_std = np.array(  [np.sqrt(Q[3, 3]) , np.sqrt(Q[4, 4]) ,np.sqrt(Q[5, 5]) ]  )
accel_noise_std = np.array([ 20 , 20 , 20 ])
gyro_noise_std = np.array( [ 0.3, 0.3 , 0.3 ]  )
simulation_interface.simulation(100,func_acc_gt , func_gyro_gt , Q , x_gt,x_est, P,accel_noise_std ,gyro_noise_std,R_gps)
