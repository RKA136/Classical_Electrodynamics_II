import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, epsilon_0, pi

class MovingCharge:
    """Point charge moving along +x with constant velocity"""
    def __init__(self, q, v, x0=0.0):
        self.q = q
        self.v = v
        self.x0 = x0

    def xpos(self, t): return self.x0 + self.v * t
    def ypos(self, t): return 0.0
    def zpos(self, t): return 0.0

    def xvel(self, t): return self.v
    def yvel(self, t): return 0.0
    def zvel(self, t): return 0.0

    def retarded_time(self, t, X, Y, Z, tol=1e-12, maxiter=100):
        """Compute retarded time iteratively"""
        tr = t - np.sqrt((X - self.xpos(t))**2 + Y**2 + Z**2)/c
        for _ in range(maxiter):
            R = np.sqrt((X - self.xpos(tr))**2 + Y**2 + Z**2)
            tr_next = t - R / c
            if np.max(np.abs(tr_next - tr)) < tol:
                break
            tr = tr_next
        return tr

def E_field(charge, t, X, Y, Z):
    tr = charge.retarded_time(t, X, Y, Z)
    rx, ry, rz = X - charge.xpos(tr), Y, Z
    R = np.sqrt(rx**2 + ry**2 + rz**2) + 1e-20
    nx, ny, nz = rx/R, ry/R, rz/R

    beta_x, beta_y, beta_z = charge.xvel(tr)/c, 0.0, 0.0
    n_dot_beta = nx*beta_x + ny*beta_y + nz*beta_z

    denom = ((1 - n_dot_beta)**3 * R**2)/(1-beta_x**2)
    prefactor = charge.q / (4 * pi * epsilon_0)
    Ex = prefactor * (nx - beta_x) / denom
    Ey = prefactor * (ny - beta_y) / denom
    Ez = prefactor * (nz - beta_z) / denom

    return Ex, Ey, Ez


def B_field(Ex, Ey, Ez, rx, ry, rz, R):
    """Magnetic field using B = (R̂ × E) / c"""
    nx, ny, nz = rx/R, ry/R, rz/R
    Bx = (ny*Ez - nz*Ey)/c
    By = (nz*Ex - nx*Ez)/c
    Bz = (nx*Ey - ny*Ex)/c
    return Bx, By, Bz


q = 1.0e-9
velocities = [0.1*c, 0.9*c, 0.999*c]
t = 0.0

# Grid for XY-plane (E-field)
x = np.linspace(-1, 1, 50) * 1e-6
y = np.linspace(-1, 1, 50) * 1e-6
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Grid for YZ-plane (B-field)
y_plane = np.linspace(-1, 1, 50) * 1e-6
z_plane = np.linspace(-1, 1, 50) * 1e-6
Yp, Zp = np.meshgrid(y_plane, z_plane)
X_plane = np.zeros_like(Yp)

#  Electric Field Plots
figE, axsE = plt.subplots(1, 3, figsize=(15, 5))
figE.suptitle("Electric Field (XY-plane)", fontsize=14)

E_log_min, E_log_max = None, None
E_mags_all = []

for v in velocities:
    charge = MovingCharge(q, v)
    Ex, Ey, Ez = E_field(charge, t, X, Y, Z)
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    E_mags_all.append(E_mag)

E_log_min = np.log10(np.min(E_mags_all))
E_log_max = np.log10(np.max(E_mags_all))

for i, v in enumerate(velocities):
    charge = MovingCharge(q, v)
    Ex, Ey, Ez = E_field(charge, t, X, Y, Z)
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    Ex_norm, Ey_norm = Ex/E_mag, Ey/E_mag
    E_log = np.log10(E_mag)

    ax = axsE[i]
    skip = (slice(None, None, 2), slice(None, None, 2))
    qv = ax.quiver(X[skip]*1e6, Y[skip]*1e6, Ex_norm[skip], Ey_norm[skip],
                   E_log[skip], cmap='plasma', scale=20, clim=(E_log_min, E_log_max))
    ax.set_title(f"v = {v/c:.4f} c")
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("y (μm)")
    ax.set_aspect('equal')

cbarE = figE.colorbar(qv, ax=axsE, orientation='horizontal', fraction=0.05, pad=0.1)
cbarE.set_label("log10(|E|)")
plt.savefig('Assignment_2/Question_1/figures/Electric_Field.png', dpi=300)
plt.show()

# Magnetic Field Plots
figB, axsB = plt.subplots(1, 3, figsize=(15, 5))
figB.suptitle("Magnetic Field (YZ-plane)", fontsize=14)

B_log_min, B_log_max = None, None
B_mags_all = []

for v in velocities:
    charge = MovingCharge(q, v)
    Ex, Ey, Ez = E_field(charge, t, X_plane, Yp, Zp)
    E_mag = np.sqrt(Ex**2 + Ey**2 + Ez**2)
    rx, ry, rz = -charge.xpos(t), Yp, Zp
    R = np.sqrt(rx**2 + ry**2 + rz**2)
    Bx, By, Bz = B_field(Ex, Ey, Ez, rx, ry, rz, R)
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    B_mags_all.append(B_mag)

B_log_min = np.log10(np.min(B_mags_all))
B_log_max = np.log10(np.max(B_mags_all))

for i, v in enumerate(velocities):
    charge = MovingCharge(q, v)
    Ex, Ey, Ez = E_field(charge, t, X_plane, Yp, Zp)
    rx, ry, rz = -charge.xpos(t), Yp, Zp
    R = np.sqrt(rx**2 + ry**2 + rz**2)
    Bx, By, Bz = B_field(Ex, Ey, Ez, rx, ry, rz, R)
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    By_norm, Bz_norm = By/B_mag, Bz/B_mag
    B_log = np.log10(B_mag)

    ax = axsB[i]
    skip = (slice(None, None, 2), slice(None, None, 2))
    qv = ax.quiver(Yp[skip]*1e6, Zp[skip]*1e6, By_norm[skip], Bz_norm[skip],
                   B_log[skip], cmap='inferno', scale=20, clim=(B_log_min, B_log_max))
    ax.set_title(f"v = {v/c:.4f} c")
    ax.set_xlabel("y (μm)")
    ax.set_ylabel("z (μm)")
    ax.set_aspect('equal')

cbarB = figB.colorbar(qv, ax=axsB, orientation='horizontal', fraction=0.05, pad=0.1)
cbarB.set_label("log10(|B|)")
plt.savefig('Assignment_2/Question_1/figures/Magnetic_Field.png', dpi=300)
plt.show()
