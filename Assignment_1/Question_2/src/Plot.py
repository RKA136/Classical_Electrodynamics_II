import numpy as np
import matplotlib.pyplot as plt

velocities = [0.01, 0.5, 0.9999]
x = np.linspace(-10, 10, 400)

# Plotting
plt.figure(figsize=(25 , 8))

for i, v in enumerate(velocities):
    # Light cone lines
    t_light_cone_pos = x
    t_light_cone_neg = -x
    
    # t' axis
    t_prime = x/v if v!=0 else np.full_like(x, np.nan)
    
    # x' axis
    x_prime = v*x
    
    plt.subplot(1, 3, i+1)
    plt.plot(t_light_cone_pos, x, label="Light Cone (Forward)", color="blue", alpha=0.5)
    plt.plot(t_light_cone_neg, x, label="Light Cone (Backward)", color="blue", alpha=0.5)
    plt.plot(x, t_prime, label="t' axis", color="red", ls='-')
    plt.plot(x, x_prime, label="x' axis", color="green", ls='--')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.axhline(0, color='black',linewidth=0.5, ls='--')
    plt.axvline(0, color='black',linewidth=0.5, ls='--')
    plt.title(f'Velocity v = {v}c')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.savefig('Assignment_1/Question_2/figures/Lorentz_transformation.png')

plt.figure(figsize=(25 , 8))
# for it and x axis
for i, v in enumerate(velocities):    
    # it' axis
    it_prime = - x/v if v!=0 else np.full_like(x, np.nan)
    
    # x' axis
    x_prime = v*x
    
    plt.subplot(1, 3, i+1)
    plt.plot(x, it_prime, label="it' axis", color="red", ls='-')
    plt.plot(x, x_prime, label="x' axis", color="green", ls='--')
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)
    plt.axhline(0, color='black',linewidth=0.5, ls='--')
    plt.axvline(0, color='black',linewidth=0.5, ls='--')
    plt.title(f'Velocity v = {v}c')
    plt.xlabel('x')
    plt.ylabel('it')
    plt.grid()
    plt.legend()
plt.tight_layout()
plt.savefig('Assignment_1/Question_2/figures/it_x_axes.png')
