#!/usr/bin/env python

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

def plot_trajectory_with_axes(positions, rotations, axis_length=0.1):
    """
    Plot 3D trajectory with orientation axes.

    Parameters:
    - positions: Nx3 array of 3D positions
    - rotations: Nx3x3 array of rotation matrices corresponding to each position
    - axis_length: length of each axis drawn
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the trajectory
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Trajectory', color='gray')

    for pos, R in zip(positions, rotations):
        # Origin of the axes
        origin = pos

        # Extract rotated axes
        x_axis = R[:, 0] * axis_length
        y_axis = R[:, 1] * axis_length
        z_axis = R[:, 2] * axis_length * 0.1

        # Plot axes as quivers
        ax.quiver(*origin, *x_axis, color='r', arrow_length_ratio=0.0, linewidth=1)
        ax.quiver(*origin, *y_axis, color='g', arrow_length_ratio=0.0, linewidth=1)
        ax.quiver(*origin, *z_axis, color='b', arrow_length_ratio=0.0, linewidth=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Trajectory with Orientation Axes')
    ax.legend()
    ax.set_box_aspect([1,1,1])
    plt.show()
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

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_3d(x_est , y_est , z_est , x_gt , y_gt, z_gt):
# Generate data for trajectory 1 (helix)
    #t1 = np.linspace(0, 10 * np.pi, 500)
    #x1 = np.sin(t1)
    #y1 = np.cos(t1)
    #z1 = t1
    
    # Generate data for trajectory 2 (offset helix)
    #t2 = np.linspace(0, 10 * np.pi, 500)
    #x2 = np.sin(t2 + np.pi / 4)
    #y2 = np.cos(t2 + np.pi / 4)
    #z2 = t2
    
    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot both trajectories
    ax.plot(x_gt, y_gt, z_gt, label='Trajectory Ground Truth', linewidth=2)
    ax.plot(x_est, y_est, z_est, label='Trajectory Estimated', linewidth=2, linestyle='--')
    
    # Labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Two 3D Trajectories')
    ax.legend()
    
    plt.show()

#def simulation(sim_time,func_gt_acc,func_gt_gyro , Q , x_gt ,x_est, P):
def simulation(sim_time,func_gt_acc,func_gt_gyro , Q , x_gt,x_est,P,accel_noise_std,gyro_noise_std,R_gps):
# --- Simulation parameters ---
    dt = 0.01
    #sim_time = 30
    steps = int(sim_time / dt)
    gyro_bias = x_gt[4]  # np.array([0.01, -0.02, 0.005]) # GT
    accel_bias = x_gt[3]  # np.array([2.0, -0.1, 3.05]) # GT 
    
    # --- Trajectory simulation ---
    pos_true = np.zeros((3, steps))
    vel_true = np.zeros((3, steps))
    att_true = np.zeros((3, steps))
    b_a_true = np.zeros((3, steps))
    b_g_true = np.zeros((3, steps))
    acc_true = np.zeros((3, steps))
    gyro_true = np.zeros((3, steps))
    
    Q_dummy = np.identity(12) * 0.000001
    P_dummy = np.identity(15) * 0.000001
    
    p , v, R , b_a,b_g= x_gt
    phi , la ,h = p 
    v_n , v_e , v_d = v

    rotations = []
    for k in range(steps):
        t = k * dt
        b_a_true[:, k] = accel_bias
        b_g_true[:,k] = gyro_bias
        #b_a_true[:, k] = [0,0,0]
        #b_g_true[:,k] =[0,0,0]
        # func_gt_acc,func_gt_gyro 
        #gyro_true[:,k] = func_gyro_gt(t)
        #acc_true[:,k]= func_acc_gt(t)
        gyro_true[:,k] = func_gt_gyro(t)
        acc_true[:,k]= func_gt_acc(t)
        f_ib_b = np.array(acc_true[:,k])
        omega_ib_b = np.array(gyro_true[:,k])
        
        #b_a = np.array([0,0,0]) # In GT No bias is included ! 
        #b_g = np.array([0,0,0]) # In GT No bias is included !!! 
        phi , la , h ,  v_n , v_e , v_d , R , _  = prediction.prediction(dt ,P_dummy,Q_dummy,
                                                                             phi , la , h,
                                                                             v_n,v_e,v_d,
                                                                             R ,
                                                                             f_ib_b , omega_ib_b,b_a,b_g)
        #r = Rot.from_matrix(R)
        if k % 100 == 0:
            rotations.append(R)
            pass
        #att_true[:,k] = np.array(r.as_euler('xyz'))
        att_true[:,k] = np.array([ np.arctan2(R[2,1],R[2,2]) , -np.arcsin(R[2,0]) ,np.arctan2(R[1,0],R[0,0])   ])
        pos_true[0, k] = phi
        pos_true[1, k] = la
        pos_true[2, k] = h
        vel_true[0, k] = v_n
        vel_true[1, k] = v_e
        vel_true[2, k] = v_d
        b_a_true[:, k] = b_a 
        b_g_true[:,k] = b_g


    # --- IMU simulation ---
    #gyro_noise_std = 0.002
    #accel_noise_std = 100.0
    
    rotations = np.array(rotations)
    imu_gyro = gyro_true - gyro_bias[:, None] + np.random.randn(3, steps) * gyro_noise_std[:, None]
    imu_accel = acc_true  - accel_bias[:, None] + np.random.randn(3, steps) * accel_noise_std[:, None]
    
    # --- State initialization ---
    x_nom = np.zeros((15, steps))  # position, velocity, orientation (not implemented)
    p , v, R , b_a,b_g= x_est
    x_nom[:,0] = np.concatenate(( p, v , [0,0,0] , b_a,b_g   ), axis=None)
    
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
    
    
    #R_gps = np.eye(3) * 0.009  # 3 m std dev
    #R_gps[0,0] =0.0000000001  # 3 m std dev
    #R_gps[1,1] =0.0000000001  # 3 m std dev
    I15 = np.eye(15)
    
    # --- GPS data ---
    gps_rate = 1.0
    gps_steps = int(1 / dt * gps_rate)
    gps_data = pos_true + np.random.randn(3, steps) * 0.000003
    gps_available = np.zeros(steps, dtype=bool)
    gps_available[::gps_steps] = True
    
    
    
    
    # --- ESKF loop ---
    for k in range(1, steps):
        #for k in range(1, 2):
        f_ib_b = imu_accel[:, k]
        omega_ib_b = imu_gyro[:, k]
        phi,la,h =  x_nom[0:3, k-1]
        v_n,v_e,v_d = x_nom[3:6, k-1]
        b_a = x_nom[9:12, k-1]
        b_g = x_nom[12:15, k-1] 
        phi , la , h ,  v_n , v_e , v_d , R , P  = prediction.prediction(dt ,P,Q,
                                                                         phi , la , h,
                                                                         v_n,v_e,v_d,
                                                                         R ,
                                                                         f_ib_b , omega_ib_b,b_a,b_g)
        x_nom[0:3, k] = [phi,la ,h]
        x_nom[3:6, k] = [v_n,v_e,v_d]
        x_nom[6:9,k] = [ np.arctan2(R[2,1],R[2,2]) , -np.arcsin(R[2,0]) ,np.arctan2(R[1,0],R[0,0])   ]
        x_nom[9:12, k] = b_a
        x_nom[12:15, k] = b_g

        if gps_available[k]:
            z = gps_data[:, k]
            h = x_nom[0:3, k]
            y = z - h
            H = np.zeros((3, 15))
            H[0:3, 0:3] = np.eye(3)
            S = H @ P @ H.T + R_gps
            K = P @ H.T @ np.linalg.inv(S)
            delta_x = K @ y
            x_nom[0:3, k] += delta_x[0:3] # Postion
            x_nom[3:6, k] += delta_x[3:6] # Vel 
            up_att = strapdown.skew_symmetric(delta_x[6:9]) # P matrix in Farrel
            #R = (np.identity(3)+ up_att) @ R
            R = alg.expm(up_att) @ R
            #print(np.linalg.det(R))
            b_a += delta_x[9:12]
            b_g += delta_x[12:15]
            P = (I15 - K @ H) @ P

        #r = Rot.from_matrix(R)
        #x_nom[6:9, k] = np.array(r.as_euler('xyz'))
        x_nom[6:9,k] = [ np.arctan2(R[2,1],R[2,2]) , -np.arcsin(R[2,0]) ,np.arctan2(R[1,0],R[0,0])   ]
        x_nom[9:12, k] = b_a
        x_nom[12:15, k] = b_g
        #print(b_a)

        errors_pos[:, k] = x_nom[0:3, k] - pos_true[:, k]
        errors_vel[:, k] = x_nom[3:6, k] - vel_true[:, k]
        errors_att[:, k] = x_nom[6:9, k] - att_true[:, k]
        #errors_xa[:, k] = x_nom[9:12, k] - b_a_true[:, k]
        #errors_xg[:, k] = x_nom[12:15, k] - b_g_true[:, k]
        errors_xa[:, k] = b_a - b_a_true[:, k]
        errors_xg[:, k] = b_g - b_g_true[:, k]
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
    #print(b_a)
    #print(b_g)
    draw_3d(x_nom[0] , x_nom[1] , x_nom[2] , pos_true[0] , pos_true[1], pos_true[2])


    
    import matplotlib.pyplot as plt
    
    # Prepare data for plotting
    time = np.arange(steps) * dt
    state_labels = ['Position', 'Velocity', 'Orientation Error Euler ZYX', 'Accel Bias', 'Gyro Bias']
    
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


    state_labels_axis = [['Latitude[ Rad ]','Longitude[ Rad ]','Height[m]'],
                    ['V_n[m/s]','V_e[m/s]','V_d[m/s]'],
                     ['Pitch[ Rad ]','Roll[ Rad ]','Yaw[Rad]'],
                     ['x[m/s^2]','y[m/s^2]','z[m/s^2]'],
                     ['x[ Rad/s ]','y[ Rad/s ]','z[ Rad/s ]']]


    
    # Create figure windows
    for state_index in range(5):
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        fig.suptitle(f'{state_labels[state_index]} State Errors and 2σ Bounds')
        #axes_labels = ['X (North)', 'Y (East)', 'Z (Down)']
        axes_labels = state_labels_axis[state_index]

        for i in range(3):
            axs[i].plot(time, state_errors[state_index, i], label='Error')
            axs[i].plot(time, 1 * np.sqrt(state_covs[state_index, i]), 'r--', label='+σ')
            axs[i].plot(time, -1 * np.sqrt(state_covs[state_index, i]), 'r--', label='-σ')
            axs[i].set_ylabel(f'{axes_labels[i]}')
            axs[i].grid(True)
            if i == 2:
                axs[i].set_xlabel('Time [s]')
            axs[i].legend()

        #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
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
    v_est = np.array([ 14-40,41-100,4+100])
    R_S = strapdown.skew_symmetric([1.2,-0.4,-6.3])
    #R_est = np.identity(3)
    R_est = alg.expm(R_S)
    b_a_est = np.array([400.5, 100.2, -1000.5])
    #b_a_est = np.array([40.5, -100.2, -1000.5])
    b_g_est = np.array([10.01, -10.02, 1.105])
    #b_a_est = np.array([10.5, -10.2, -2.5])
    #b_g_est = np.array([1.0, -0.20, 0.5])
    x_est = (p_est,v_est , R_est , b_a_est , b_g_est)
    P = np.eye(15) * 1.0
    P[0,0] *= 0.000000003
    P[1,1] *= 0.000000003
    P[3:6, 3:6] *= 100
    P[9:12, 9:12] *= 1000**2
    P[12:15, 12:15] *= 5**2
    
    R_gps = np.eye(3) * 0.00009  # 3 m std dev
    R_gps[0,0] =0.0000000001  # 3 m std dev
    R_gps[1,1] =0.0000000001  # 3 m std dev
    R_gps[2,2] = 0.0000009  # 3 m std dev
    #simulation(30,func_acc_gt , func_gyro_gt , Q , x_gt ,x_est, P)
    #accel_noise_std = np.array([np.sqrt(Q[0, 0]) , np.sqrt(Q[1, 1]) ,np.sqrt(Q[2, 2]) ])
    #gyro_noise_std = np.array(  [np.sqrt(Q[3, 3]) , np.sqrt(Q[4, 4]) ,np.sqrt(Q[5, 5]) ]  )
    accel_noise_std = np.array([ 20 , 20 , 20 ])
    gyro_noise_std = np.array( [ 0.3, 0.3 , 0.3 ]  )
    simulation(50,func_acc_gt , func_gyro_gt , Q , x_gt,x_est, P,accel_noise_std ,gyro_noise_std,R_gps)
