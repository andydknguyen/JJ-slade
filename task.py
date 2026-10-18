from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np
import matplotlib.pyplot as plt

# Beam dimensions
L_0 = 0.2 # [m]
H_0 = 0.02 # [m]
W_0 = 1 # [m]

# Material parameters
mu = 0.715 * 10**6 # [Pa]
kappa = 3.25 * 10**6 # [Pa]
lambda_ = kappa - (2/3)*mu
rho = 1000 # [kg/m^3]
g = 9.8 # [m/s^2]

# Geometry
domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0,0,0]), np.array([L_0,H_0,W_0])],
    [20,6,6],
    cell_type=mesh.CellType.hexahedron
)

V = fem.functionspace(domain, ("Lagrange",1,(domain.geometry.dim,)))

def clamped_boundary(x):
    return np.isclose(x[0],0)

def tip(x):
    return np.isclose(x[0],L_0) & np.isclose(x[1],0) & np.isclose(x[2],0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain,fdim,clamped_boundary)
u_D = np.array([0,0,0], dtype=default_scalar_type)
dofs = fem.locate_dofs_topological(V,fdim,boundary_facets)
bcs = fem.dirichletbc(u_D,dofs,V)

bf2 = mesh.locate_entities_boundary(domain,fdim,tip)
dofs2 = fem.locate_dofs_topological(V,fdim,bf2)

f = fem.Constant(domain,default_scalar_type((0,0,-rho*g)))
T = fem.Constant(domain,default_scalar_type((0,0,0)))
ds = ufl.Measure("ds",domain=domain)

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

# Declare functions
d = fem.Function(V) # displacement at t+1
d_old = fem.Function(V) # displacement at t
vel_old = fem.Function(V) # velocity at t
a_old = fem.Function(V) # acceleration at t

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a_new = fem.Function(V)
vel_new = fem.Function(V)

# Average acceleration, implicit scheme, unconditional stability
beta = 0.25
gamma = 0.5
dt = fem.Constant(domain,default_scalar_type(0))

a = (1/(beta * dt**2)) * (d - d_old - dt * vel_old) + a_old * ((1 - 2*beta)/(2*beta))
a_expr = fem.Expression(a, V.element.interpolation_points)

vel = vel_old + dt * ((1 - gamma) * a_old + gamma * a)
vel_expr = fem.Expression(vel, V.element.interpolation_points)

# Weak form representations
def mass(d,v):
    return rho * ufl.dot(d,v) * ufl.dx

def stiffness(d,v):
    return ufl.inner(sigma(d),epsilon(v)) * ufl.dx

def body_force(v):
    return ufl.dot(f,v) * ufl.dx

def traction(v):
    return ufl.dot(T,v) * ufl.dx

r = mass(a,v) + stiffness(d,v) - body_force(v) - traction(v)

res = ufl.replace(r, {d: u})
a_form = ufl.lhs(res)
L_form = ufl.rhs(res)

# Define energies
ke = fem.form(0.5 * mass(vel_old, vel_old))
pe = fem.form(0.5 * stiffness(d_old, d_old))

# Define problem
problem = LinearProblem(
    a_form, L_form, u=d, bcs=[bcs], petsc_options={"ksp_type": "preonly", "pc_type": "lu"},petsc_options_prefix="linear_elasticity_"
)

# Set up looping parameters
steps = 100
time = np.linspace(0,10,steps+1)
energy = np.zeros((steps+1,2))
u_tip = np.zeros((steps+1,3))

# Solve problem
vtk = io.VTKFile(domain.comm, "results/elastodynamics.pvd", "w")
t = 0.0

for i, dti in enumerate(np.diff(time)):
    dt.value = dti
    t += dti

    vtk.write_function(d,t)
    problem.solve()

    d.x.scatter_forward()

    vel_new.interpolate(vel_expr)
    a_new.interpolate(a_expr)
    vel_new.x.scatter_forward()
    a_new.x.scatter_forward()

    d_old.x.array[:] = d.x.array
    vel_old.x.array[:] = vel_new.x.array
    a_old.x.array[:] = a_new.x.array

    energy[i+1, 0] = fem.assemble_scalar(pe)
    energy[i+1, 1] = fem.assemble_scalar(ke)
    #u_tip[i+1, :] = d.x.array[dofs2]

vtk.close()

plt.plot(time, energy[:, 0], label="Elastic")
plt.plot(time, energy[:, 1], label="Kinetic")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Energies")
plt.savefig("energy.png")