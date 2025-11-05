from scipy.constants import epsilon_0, c, e, pi
import numpy as np
import matplotlib.pyplot as plt

V_0 = 0.5*c

def F_rad(t):
    return (e**2*V_0)/(6*pi*epsilon_0*c**3)*((1e-4-900)*np.sin(30*t)+(3/5)*np.cos(30*t))*np.exp(t/100)

t = np.linspace(0,1000,1000)

plt.figure(figsize=(10,8))
plt.plot(t,F_rad(t))
plt.xlabel('time (s)')
plt.ylabel('F_rad(t)')
plt.title('Radiation Reaction Force vs Time')
plt.grid(True)
plt.tight_layout()
plt.savefig('Assignment_2/Question_5/figures/Radiation_Force_vs_time.png', dpi=300)
plt.show()