#!/usr/bin/env python

#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import strapdown
import prediction
from scipy.spatial.transform import Rotation as Rot
import scipy.linalg as alg

# --- WGS-84 constants ---
a = 6378137.0  # semi-major axis [m]
f = 1 / 298.257223563
b = a * (1 - f)
e2 = 1 - (b**2 / a**2)

def gravity(latitude_deg: float, height: float) -> float:
    """Compute gravity at given latitude and height."""
    lat = np.radians(latitude_deg)
    sin_lat = np.sin(lat)
    g0 = 9.7803253359 * (1 + 0.00193185265241 * sin_lat**2) / np.sqrt(1 - e2 * sin_lat**2)
    g = g0 - (3.0877e-6 - 0.004e-6 * sin_lat**2) * height + 0.072e-12 * height**2
    return g

def radii_of_curvature(latitude_deg: float) -> tuple:
    """Return meridian (M) and prime vertical (N) radii of curvature."""
    lat = np.radians(latitude_deg)
    sin_lat = np.sin(lat)
    denom = np.sqrt(1 - e2 * sin_lat**2)
    N = a / denom
    M = a * (1 - e2) / denom**3
    return M, N

# --- Simulation parameters ---
dt = 0.01
sim_time = 30
steps = int(sim_time / dt)
latitude = np.pi / 4
altitude = 500.0

phi  = latitude
la = 0 
h = altitude

# Gravity and curvature at given location
#g_val = gravity(phi, h)
#g_ned = np.array([0, 0, g_val])
M, N = radii_of_curvature(phi)

# --- Trajectory simulation ---
radius = 50000
omega = 2 * np.pi / 60
pos_true = np.zeros((3, steps))
vel_true = np.zeros((3, steps))
att_true = np.zeros((3, steps))
b_a_true = np.zeros((3, steps))
b_g_true = np.zeros((3, steps))
acc_true = np.zeros((3, steps))
gyro_true = np.zeros((3, steps))

gyro_bias = np.array([0.01, -0.02, 0.005])
accel_bias = np.array([0.2, -0.1, 0.05])

#print("b_a GT :",accel_bias)

R = np.identity(3)
p = np.array([np.pi /4 , np.pi/4 , 40])
v = np.arange(3)
v[0] = 0.0
v[1] = 0.0
v[2] = 0.0
b_a = np.array([10.5, -10.2, -2.5])
#print("b_a before is :",b_a)
b_g = np.array([0.01, -0.02, 0.005])
f_ib_b = np.arange(3)
omega_ib_b = np.array([0.0,0.0,0.0])
dt = 0.01 
phi , la ,h = p 
v_n , v_e , v_d = v
#Q = np.arange(144).reshape(12,12)
Q = np.identity(12) * 0.000001
#P = np.arange(15*15).reshape(15,15)
P = np.identity(15) * 0.000001

R = np.identity(3)
p = np.array([np.pi /4 , np.pi/4 , 0])
v = np.array([ 14,41,4])
v = np.array([ 0 ,0 ,0])
#b_a = np.array([0.2, -0.1, 0.05])
#b_g = np.array([0.01, -0.02, 0.005])
#f_ib_b = np.arange(3)
#omega_ib_b = np.array([0,0,0])
dt = 0.01 
phi , la ,h = p 
v_n , v_e , v_d = v

for k in range(steps):
    t = k * dt
    b_a_true[:, k] = accel_bias
    b_g_true[:,k] = gyro_bias
    #acc_true[0, k] = -radius * omega**2 * np.cos(omega * t)
    acc_true[0, k] = 0.0
    #
    #acc_true[1, k] = -radius * omega**2 * np.sin(omega * t)
    acc_true[1, k] = 0.0
    acc_true[2, k] = -9.81
    gyro_true[:,k] = omega_ib_b
    f_ib_b = np.array(acc_true[:,k])
    phi , la , h ,  v_n , v_e , v_d , R , P_new  = prediction.prediction(dt ,P,Q,
                                                                         phi , la , h,
                                                                         v_n,v_e,v_d,
                                                                         R ,
                                                                         f_ib_b , omega_ib_b,b_a,b_g)
    r = Rot.from_matrix(R)
    att_true[:,k] = np.array(r.as_euler('xyz'))
    pos_true[0, k] = phi
    pos_true[1, k] = la
    pos_true[2, k] = h
    vel_true[0, k] = v_n
    vel_true[1, k] = v_e
    vel_true[2, k] = v_d
    b_a_true[:, k] = b_a 
    b_g_true[:,k] = b_g


# --- IMU simulation ---
gyro_noise_std = 0.002
accel_noise_std = 100.0

#imu_gyro = gyro_true - gyro_bias[:, None] + np.random.randn(3, steps) * gyro_noise_std
imu_gyro = gyro_true - gyro_bias[:, None]
imu_accel = acc_true  - accel_bias[:, None] + np.random.randn(3, steps) * accel_noise_std

# --- State initialization ---
x_nom = np.zeros((15, steps))  # position, velocity, orientation (not implemented)
#x_nom[:,0] = [np.pi /4 + 0.0001 , np.pi/4 + 0.0001 , 0,
#              14 + 20 ,41 - 10,4 + 10,
#              0,0,0,
#              0,0,0,
#              0,0,0,]
x_nom[:,0] = [np.pi /4 + 0.0001 , np.pi/4 + 0.0001 , 0,
                0 ,0,0,
              0,0,0,
              0,0,0,
              0,0,0,]
#x_est = np.zeros((15, steps))  # full ESKF state
#x_est[:,0] = [np.pi /4 , np.pi/4 , 0,
#              14,41,4,
#              4,5,1,
#              3,4,5,
#              5,6,6]
P = np.eye(15) * 1.0
P[0,0] *= 0.000000003
P[1,1] *= 0.000000003
#P[0:3, 0:3] *= 0.0004
P[3:6, 3:6] *= 100
#P[6:9, 6:9] *= 1e-6
P[9:12, 9:12] *= 100
P[12:15, 12:15] *= 1

errors_pos = np.zeros((3, steps))
errors_vel = np.zeros((3, steps))
errors_att = np.zeros((3, steps))
errors_xa = np.zeros((3, steps))
errors_xg = np.zeros((3, steps))
cov_diag_pos = np.zeros((3, steps))
cov_diag_vel = np.zeros((3, steps))
cov_diag_att = np.zeros((3, steps))
cov_diag_xa = np.zeros((3, steps))
cov_diag_xg = np.zeros((3, steps))

# --- Process and measurement noise ---
Q = np.eye(12)
#Q[0:3, 0:3] *= 0.0004
#Q[3:6, 3:6] *= 0.04
#Q[6:9, 6:9] *= 1e-6
#Q[9:12, 9:12] *= 1e-4

Q[0, 0] *= 0.02
Q[1, 1] *= 0.02
Q[2, 2] *= 0.04
#Q[3:6, 3:6] *= 0.004
Q[3, 3] *= 0.0000004
Q[4, 4] *= 0.0000004
Q[5, 5] *= 0.04
#Q[6:9, 6:9] *= 1e-9
Q[6:9, 6:9] *= 1e-2
Q[9:12, 9:12] *= 1e-9

R_gps = np.eye(3) * 0.009  # 3 m std dev
R_gps[0,0] =0.0000000001  # 3 m std dev
R_gps[1,1] =0.0000000001  # 3 m std dev
I15 = np.eye(15)

# --- GPS data ---
gps_rate = 2.0
gps_steps = int(1 / dt * gps_rate)
gps_data = pos_true + np.random.randn(3, steps) * 0.000003
gps_available = np.zeros(steps, dtype=bool)
gps_available[::gps_steps] = True




# --- ESKF loop ---
for k in range(1, steps):
#for k in range(1, 2):
    f_ib_b = imu_accel[:, k]
    omega_ib_b = imu_accel[:, k]
    phi,la,h =  x_nom[0:3, k-1]
    v_n,v_e,v_d = x_nom[3:6, k-1]
    b_a = x_nom[9:12, k-1]
    b_g = x_nom[12:15, k-1] 
    #print(phi)
    #print(P[0,0])
    
     
    phi , la , h ,  v_n , v_e , v_d , R , P  = prediction.prediction(dt ,P,Q,
                                                                         phi , la , h,
                                                                         v_n,v_e,v_d,
                                                                         R ,
                                                                         f_ib_b , omega_ib_b,b_a,b_g)
    #if (steps//4) % 100:
        #print(phi)

    
    #vel = x_nom[3:6, k-1] + acc_meas * dt
    #pos = x_nom[0:3, k-1] + x_nom[3:6, k-1] * dt + 0.5 * acc_meas * dt**2
    x_nom[0:3, k] = [phi,la ,h]
    x_nom[3:6, k] = [v_n,v_e,v_d]
    x_nom[9:12, k] = b_a
    x_nom[12:15, k] = b_g


    #F = np.eye(15)
    #F[0:3, 3:6] = np.eye(3) * dt
    #F[3:6, 6:9] = -np.eye(3) * dt
    #F[3:6, 12:15] = -np.eye(3) * dt

    #G = np.zeros((15, 12))
    #G[6:9, 0:3] = -np.eye(3)
    #G[3:6, 3:6] = -np.eye(3)
    #G[9:12, 6:9] = np.eye(3)
    #G[12:15, 9:12] = np.eye(3)

    #P = F @ P @ F.T + G @ Q @ G.T

    if gps_available[k]:
        #print("GPS")
        z = gps_data[:, k]
        h = x_nom[0:3, k]
        y = z - h
        H = np.zeros((3, 15))
        H[0:3, 0:3] = np.eye(3)
        S = H @ P @ H.T + R_gps
        K = P @ H.T @ np.linalg.inv(S)
        delta_x = K @ y
        x_nom[0:3, k] += delta_x[0:3]
        x_nom[3:6, k] += delta_x[3:6]
        up_att = strapdown.skew_symmetric(delta_x[6:9]) # P matrix in Farrel
        #R = (np.identity(3)+ up_att) @ R
        #R = alg.expm(up_att) @ R
        print(np.linalg.det(R))
        b_a += delta_x[9:12]
        b_g += delta_x[12:15]
        P = (I15 - K @ H) @ P

    r = Rot.from_matrix(R)
    x_nom[6:9, k] = np.array(r.as_euler('xyz'))
    x_nom[9:12, k] = b_a
    x_nom[12:15, k] = b_g

    errors_pos[:, k] = x_nom[0:3, k] - pos_true[:, k]
    errors_vel[:, k] = x_nom[3:6, k] - vel_true[:, k]
    errors_att[:, k] = x_nom[6:9, k] - att_true[:, k]
    errors_xa[:, k] = x_nom[9:12, k] - b_a_true[:, k]
    errors_xg[:, k] = x_nom[12:15, k] - b_g_true[:, k]
    #print(errors_pos[:, k])
    cov_diag_pos[:, k] = np.diag(P)[0:3]
    cov_diag_vel[:, k] = np.diag(P)[3:6]
    cov_diag_att[:, k] = np.diag(P)[6:9]
    cov_diag_xa[:, k] = np.diag(P)[9:12]
    cov_diag_xg[:, k] = np.diag(P)[12:15]



# --- Plotting ---
#import ace_tools_open as tools; tools.display_dataframe_to_user(name="Position Error and Covariance", dataframe=
#    pd.DataFrame({
#        'Time [s]': np.arange(steps) * dt,
#        'North Error [m]': errors_pos[0],
#        'East Error [m]': errors_pos[1],
#        'Down Error [m]': errors_pos[2],
#        'North 2σ [m]': 2*np.sqrt(cov_diag_pos[0]),
#        'East 2σ [m]': 2*np.sqrt(cov_diag_pos[1]),
#        'Down 2σ [m]': 2*np.sqrt(cov_diag_pos[2])
#    })
#)
import matplotlib.pyplot as plt

# Prepare data for plotting
time = np.arange(steps) * dt
state_labels = ['Position', 'Velocity', 'Orientation Error Euler ZYX', 'Accel Bias', 'Gyro Bias']
state_labels_axis = {'Position':['Rad','Rad','m']
                     , 'Velocity':['m/s','m/s','m/s'],
                     'Orientation Error Euler ZYX': ['Rad','Rad','Rad'],
                     'Accel Bias': ['m/s^2','m/s^2','m/s^2'],
                     'Gyro Bias': ['Rad','Rad','Rad']}
state_labels_axis = [['[ Rad ]','[ Rad ]','[ m ]'],
                    ['[m/s]','[m/s]','[m/s]'],
                     ['[ Rad ]','[ Rad ]','[ Rad ]'],
                     ['[m/s^2]','[m/s^2]','[m/s^2]'],
                     ['[ Rad ]','[ Rad ]','[ Rad ]']]


# Simulated state errors and covariance (only position used so far, so we fill others with dummy zeros)
state_errors = np.zeros((5, 3, steps))
state_covs = np.zeros((5, 3, steps))

# Fill in position data
state_errors[0] = errors_pos
state_covs[0] = cov_diag_pos

state_errors[1] = errors_vel
state_covs[1] = cov_diag_vel

state_errors[2] = errors_att
state_covs[2] = cov_diag_att

state_errors[3] = errors_xa
state_covs[3] = cov_diag_xa

state_errors[4] = errors_xg
state_covs[4] = cov_diag_xa
# Create figure windows
for state_index in range(5):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    fig.suptitle(f'{state_labels[state_index]} State Errors and σ Bounds')
    axes_labels = state_labels_axis[state_index]
    #axes_labels = ['X (North)', 'Y (East)', 'Z (Down)']

    for i in range(3):
        axs[i].plot(time, state_errors[state_index, i], label='Error')
        axs[i].plot(time, 1 * np.sqrt(state_covs[state_index, i]), 'r--', label='+σ')
        axs[i].plot(time, -1 * np.sqrt(state_covs[state_index, i]), 'r--', label='-σ')
        axs[i].set_ylabel(f'{axes_labels[i]}')
        axs[i].grid(True)
        if i == 2:
            axs[i].set_xlabel('Time [s]')
        axs[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
