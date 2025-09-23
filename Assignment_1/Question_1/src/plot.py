import numpy as np
import matplotlib.pyplot as plt

# Constants
c = 3e8  # speed of light in m/s
mu0 = 4*np.pi*1e-7
eps0 = 1/(mu0*c**2)

# Random fields and charge/current (simple example)
E = np.array([1.0, 2.0, 3.0])       # Electric field
B = np.array([0.5, 1.0, 0.0])       # Magnetic field
rho = 1e-6                           # Charge density
j = np.array([1e-3, 2e-3, 0.0])     # Current density

# Assume some constant derivatives for simplicity
dE_dt = np.array([0.1, 0.05, -0.02])
dB_dt = np.array([0.02, -0.01, 0.03])
grad_E = np.array([0.05, 0.1, -0.05])
grad_B = np.array([-0.02, 0.03, 0.01])

# Velocities to consider
v_frac = np.logspace(-2, -0.0001, 100)
v_vals = v_frac * c

# Arrays to store differences
diff_Faraday = []
diff_Ampere = []

for v in v_vals:
    # --- Galilean transformations ---
    # Faraday: curl E = -dB/dt - (v . grad) B
    Faraday_gal = -dB_dt - v * grad_B
    
    # Ampere-Maxwell: curl B = mu0 j + mu0 eps0 dE/dt + mu0 eps0 (v . grad) E - mu0 rho v
    Ampere_gal = mu0*j + mu0*eps0*dE_dt + mu0*eps0*v*grad_E - mu0*rho*v
    
    # --- Lorentz transformations ---
    beta = v/c
    gamma = 1/np.sqrt(1-beta**2)
    
    # Lorentz transforms for fields along x-direction (boost along x)
    E_lor = np.zeros(3)
    B_lor = np.zeros(3)
    E_lor[0] = E[0]
    E_lor[1] = gamma*(E[1] - beta*c*B[2])
    E_lor[2] = gamma*(E[2] + beta*c*B[1])
    B_lor[0] = B[0]
    B_lor[1] = gamma*(B[1] + beta*E[2]/c)
    B_lor[2] = gamma*(B[2] - beta*E[1]/c)
    
    # Lorentz derivatives (simplified linear approximation)
    dE_dt_lor = gamma * (dE_dt + beta*c*np.cross([1,0,0], dB_dt))
    dB_dt_lor = gamma * (dB_dt - beta/c*np.cross([1,0,0], dE_dt))
    
    # Faraday: curl E = -dB/dt
    Faraday_lor = -dB_dt_lor
    # Ampere-Maxwell: curl B = mu0 j + mu0 eps0 dE/dt
    Ampere_lor = mu0*j + mu0*eps0*dE_dt_lor
    
    # Compute differences
    diff_Faraday.append(np.linalg.norm(Faraday_lor - Faraday_gal))
    diff_Ampere.append(np.linalg.norm(Ampere_lor - Ampere_gal))

# --- Plotting ---
fig, axs = plt.subplots(1, 2, figsize=(12,5))

axs[0].plot(v_frac, diff_Faraday, color='blue', linewidth=2)
axs[0].set_title("Difference in Faraday's Law")
axs[0].set_xlabel("v/c")
axs[0].set_ylabel("|Faraday_Lorentz - Faraday_Galilean|")
axs[0].grid(True, which='both', ls='--')

axs[1].plot(v_frac, diff_Ampere, color='red', linewidth=2)
axs[1].set_title("Difference in Ampere-Maxwell Law")
axs[1].set_xlabel("v/c")
axs[1].set_ylabel("|Ampere_Lorentz - Ampere_Galilean|")
axs[1].grid(True, which='both', ls='--')

plt.tight_layout()
plt.savefig('Assignment_1/Question_1/figures/field_differences.png', dpi=300)
plt.show()
