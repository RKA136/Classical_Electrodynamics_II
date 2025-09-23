import numpy as np
import matplotlib.pyplot as plt

v_list = [0.01, 0.5, 0.9999] 
v_cont = np.linspace(0.01, 0.9999, 1000)

def gamma(v):
    return 1 / np.sqrt(1 - v**2)

gamma_points = gamma(np.array(v_list))
#Proper lifetime
tau = 1  # Proper lifetime in its rest frame
tau_obs_points = tau * gamma_points  # Observed lifetime for discrete velocities
plt.figure(figsize=(10, 6))
plt.plot(v_cont, tau*gamma(v_cont), label=r'$\gamma(v)$', color='blue')
plt.scatter(v_list, tau_obs_points, color='red')
for i, v in enumerate(v_list):
    plt.annotate(f"v/c = {v:.4f}\nγ = {gamma_points[i]:.4f}", (v_list[i], tau_obs_points[i]/tau),
                 textcoords="offset points", xytext=(5,5), fontsize=9)

plt.title(r'Lifetime $\tau$ as a Function of Velocity v')
plt.xlabel('Velocity v (in units of c)')
plt.ylabel(r'Observed Lifetime $\tau_{obs} = \gamma \tau$')
plt.tight_layout()
plt.grid()
plt.savefig('Assignment_1/Question_3/figures/lifetime_vs_velocity.png')