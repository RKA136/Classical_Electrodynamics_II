import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, pi

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

q = 1e-19
r_obs = np.array([10e-6, 10e-6, 10e-6])
t_obs = np.logspace(-15,-9, 1000)

beta0_values = [0.1, 0.9, 0.99]
cases = [(0.3, 'z', 'x'),
         (0.9, 'x', 'x')]

for beta_dot, accel_axis, vel_axis in cases:
    for beta0 in beta0_values:
        charge = AcceleratedCharge(q, beta_dot, accel_axis=accel_axis, vel_axis=vel_axis, beta0=beta0)
        E, B = radiation_field(charge, r_obs, t_obs)

        # Electric Field Plot
        plt.figure(figsize=(12,8))
        plt.subplot(2,1,1)
        plt.plot(t_obs, E[:,0], linestyle='-', color='blue', alpha=0.5, label='Ex')
        plt.plot(t_obs, E[:,1], linestyle='--', color='green', alpha=0.5, label='Ey')
        plt.plot(t_obs, E[:,2], linestyle=':', color='red', alpha=0.5, label='Ez')
        plt.xlabel('Time (s, log scale)')
        plt.ylabel('Electric Field (V/m)')
        plt.title(f'E-field vs Time (beta_dot={beta_dot}, beta0={beta0}, accel={accel_axis}, vel={vel_axis})')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, which='both', linestyle='--')

        # Magnetic Field Plot
        plt.subplot(2,1,2)
        plt.plot(t_obs, B[:,0], linestyle='-', color='blue', alpha=0.5, label='Bx')
        plt.plot(t_obs, B[:,1], linestyle='--', color='green', alpha=0.5, label='By')
        plt.plot(t_obs, B[:,2], linestyle=':', color='red', alpha=0.5, label='Bz')
        plt.xlabel('Time (s, log scale)')
        plt.ylabel('Magnetic Field (T)')
        plt.title(f'B-field vs Time (beta_dot={beta_dot}, beta0={beta0}, accel={accel_axis}, vel={vel_axis})')
        plt.xscale('log')
        plt.legend()
        plt.grid(True, which='both', linestyle='--')

        plt.tight_layout()
        plt.savefig(f'Assignment_2/Question_2/figures/fields_betaDot{beta_dot}_beta0{beta0}_accel{accel_axis}_vel{vel_axis}.png')
        plt.show()
