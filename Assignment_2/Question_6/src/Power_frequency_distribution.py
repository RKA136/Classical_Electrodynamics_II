import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import h, k

# Thermal Bremsstrahlung Power Spectrum (Up to a Constant Factor)

def bremsstrahlung_power(nu, T, prefactor=1.0):
    x = (h*nu)/(k*T)
    return prefactor * T**(-0.5)*(2.2*(1+x)+0.6*(x**2))*(np.exp(-x))


# Frequency Range
nu = np.logspace(9, 20, 1200)          

# Temperatures
T1 = 300
T2 = 1e6

P1 = bremsstrahlung_power(nu, T1)
P2 = bremsstrahlung_power(nu, T2)

plt.figure(figsize=(9, 6))
plt.loglog(nu, P1, linewidth=2.0, label="T = 300 K")
plt.loglog(nu, P2, linewidth=2.0, label="T = 10⁶ K")

plt.title("Thermal Bremsstrahlung Power Spectrum\n(Arbitrary Units)", fontsize=14, fontweight="bold")
plt.xlabel("Frequency ν  [Hz]", fontsize=12)
plt.ylabel("Power per Unit Frequency  [a.u.]", fontsize=12)
plt.xlim(1e9,1e20)
plt.ylim(1e-18,1e3)
plt.grid(which="both", linestyle="--", linewidth=0.6, alpha=0.7)
plt.legend(fontsize=11, loc="upper right")
plt.tight_layout()
plt.savefig('Assignment_2/Question_6/figures/Power_vs_frequency.png', dpi = 300)
plt.show()
