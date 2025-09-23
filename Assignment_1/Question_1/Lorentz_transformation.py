import sympy as sp

# Coordinates (ct, x, y, z)
ct, x, y, z = sp.symbols('ct x y z', real=True)
c = sp.symbols('c', real=True, positive=True)

coords = [ct, x, y, z]

# Four-potential A^μ = (φ/c, A_x, A_y, A_z)
phi = sp.Function('phi')(ct, x, y, z)
Ax = sp.Function('Ax')(ct, x, y, z)
Ay = sp.Function('Ay')(ct, x, y, z)
Az = sp.Function('Az')(ct, x, y, z)

A = sp.Matrix([phi/c, Ax, Ay, Az])

# Minkowski metric diag(+,-,-,-)
eta = sp.diag(1, -1, -1, -1)

# Field strength tensor F_{μν} = ∂_μ A_ν - ∂_ν A_μ
F = sp.MutableDenseMatrix(4, 4, lambda i, j: 0)
for mu in range(4):
    for nu in range(4):
        F[mu, nu] = sp.diff(A[nu], coords[mu]) - sp.diff(A[mu], coords[nu])

# Raise indices: F^{μν} = η^{μα} η^{νβ} F_{αβ}
F_up = eta * F * eta

# Define current four-vector J^ν = (cρ, jx, jy, jz)
rho = sp.Function('rho')(ct, x, y, z)
jx = sp.Function('jx')(ct, x, y, z)
jy = sp.Function('jy')(ct, x, y, z)
jz = sp.Function('jz')(ct, x, y, z)
J = sp.Matrix([c*rho, jx, jy, jz])

# Compute ∂_μ F^{μν} (inhomogeneous Maxwell equations)
Maxwell_eqs = []
mu0 = sp.symbols('mu0')
for nu in range(4):
    divergence = sum(sp.diff(F_up[mu, nu], coords[mu]) for mu in range(4))
    Maxwell_eqs.append(sp.Eq(divergence, mu0*J[nu]))

# Display equations
for i, eq in enumerate(Maxwell_eqs):
    print(f"Inhomogeneous Maxwell equation (nu={i}):")
    sp.pprint(eq)
    print()
