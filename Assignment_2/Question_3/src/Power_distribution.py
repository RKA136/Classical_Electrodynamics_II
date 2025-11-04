import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, pi, mu_0

# Accelerated Charge Class
class AcceleratedCharge:
    def __init__(self, q, beta_dot, accel_axis='z', beta0=0.0, vel_axis='x', r0=np.zeros(3)):
        self.q = q
        self.beta_dot = beta_dot
        self.accel_axis = accel_axis
        self.vel_axis = vel_axis
        self.beta0 = beta0
        self.r0 = np.array(r0, dtype=float)

    def position(self, t):
        pos = self.r0.copy()
        delta_vel = self.beta0 * c * t
        delta_acc = 0.5 * self.beta_dot * c * t**2
        axes = {'x':0, 'y':1, 'z':2}
        pos[axes[self.vel_axis]] += delta_vel
        pos[axes[self.accel_axis]] += delta_acc
        return pos

    def velocity(self, t):
        vel = np.zeros(3)
        axes = {'x':0, 'y':1, 'z':2}
        vel[axes[self.vel_axis]] = self.beta0 * c
        vel[axes[self.accel_axis]] += self.beta_dot * c * t
        return vel / c

    def beta_dot_vec(self):
        vec = np.zeros(3)
        axes = {'x':0, 'y':1, 'z':2}
        vec[axes[self.accel_axis]] = self.beta_dot
        return vec
# Retarded time Calculation
def retarded_time(charge, r_obs, t_obs, tol=1e-15, maxiter=100):
    tr = t_obs
    for _ in range(maxiter):
        R = np.linalg.norm(r_obs - charge.position(tr))
        tr_next = t_obs - R / c
        if abs(tr_next - tr) < tol: 
            break
        tr = tr_next
    return tr
# Field Calculation
def radiation_field(charge, r_obs, t_obs):
    E_list = []
    B_list = []
    for t in t_obs:
        tr = retarded_time(charge, r_obs, t)
        r_source = charge.position(tr)
        R_vec = r_obs - r_source
        R = np.linalg.norm(R_vec)
        n = R_vec / R
        beta = charge.velocity(tr)
        beta_dot = charge.beta_dot_vec()
        k = 1 - np.dot(n, beta)
        E = (charge.q/(4*pi*epsilon_0*c)) * np.cross(n, np.cross(n - beta, beta_dot)) / (R * k**3)
        B = np.cross(n, E) / c
        E_list.append(E)
        B_list.append(B)
    return np.array(E_list), np.array(B_list)
# Angular Distribution Using Radiation Field 
def angular_radiated_power(charge, t_obs, theta_vals, phi_vals=None):
    dP = []
    for theta in theta_vals:
        phi = 0 if phi_vals is None else phi_vals
        # Observation point on unit sphere of radius R
        n_vec = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        R = 1e5
        r_obs = n_vec * R
        
        E, B = radiation_field(charge, r_obs, [t_obs])
        E = E[0]
        # S = |E|² / μ0*c
        S = np.linalg.norm(E)**2 / (mu_0*c)
        dP.append(R**2 * S)
    return np.array(dP)

theta_vals = np.linspace(0, 2*pi, 300)
t_obs_sample = 1e-13 
q = 1e-19
beta0_values = [0.1, 0.9, 0.99]
cases = [(0.3, 'z', 'x'),
         (0.9, 'x', 'x')]

for beta_dot, accel_axis, vel_axis in cases:
    for beta0 in beta0_values:
        charge = AcceleratedCharge(q, beta_dot, accel_axis=accel_axis, vel_axis=vel_axis, beta0=beta0)
        power_vals = angular_radiated_power(charge, t_obs_sample, theta_vals)
        idx = np.argmax(power_vals)
        print("peak θ (deg) =", np.degrees(theta_vals[idx]))    
        plt.figure(figsize=(8,8))
        ax = plt.subplot(111, projection='polar')
        ax.plot(theta_vals, power_vals, color='darkorange', lw=2)
        ax.set_theta_zero_location('N') 
        ax.set_theta_direction(-1)       
        ax.set_rlabel_position(135)      
        ax.set_title(f'Angular Distribution (beta_dot={beta_dot}, beta0={beta0}, accel={accel_axis})', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'Assignment_2/Question_3/figures/angular_power_polar_betaDot{beta_dot}_beta0{beta0}_accel{accel_axis}_vel{vel_axis}.png')
        plt.show()
