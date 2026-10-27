from dolfinx import mesh, fem, plot, io, default_scalar_type, geometry
from dolfinx.fem.petsc import LinearProblem, assemble_matrix, assemble_vector, apply_lifting, set_bc
from mpi4py import MPI
from petsc4py import PETSc
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

# 2D Geometry
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0,0]),np.array([L_0,H_0])],
    [12,8],
    cell_type=mesh.CellType.quadrilateral,
)

V = fem.functionspace(domain, ("Lagrange",1,(domain.geometry.dim,)))

# Boundary conditions
def clamped_boundary(x):
    return np.isclose(x[0],0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain,fdim,clamped_boundary)
u_D = np.array([0,0], dtype=default_scalar_type)
dofs = fem.locate_dofs_topological(V,fdim,boundary_facets)
bcs = fem.dirichletbc(u_D,dofs,V)

# Track cell in which top point (L_0, H_0) of beam tip is contained
bb_tree = geometry.bb_tree(domain, domain.topology.dim, padding=1e-10)
point = np.array([[L_0, H_0, 0.0]], dtype=domain.geometry.x.dtype)
potential_colliding_cells = geometry.compute_collisions_points(bb_tree, point)
colliding_cells = geometry.compute_colliding_cells(domain, potential_colliding_cells, point)

points_on_proc = []
cells = []
for i, point in enumerate(point):
    if len(colliding_cells.links(i)) > 0:
        points_on_proc.append(point)
        cells.append(colliding_cells.links(i)[0])

points_on_proc = np.array(points_on_proc, dtype=np.float64).reshape(-1, 3)
cells = np.array(cells, dtype=np.int32)

T = fem.Constant(domain,default_scalar_type((0,0)))
ds = ufl.Measure("ds", domain=domain)
f = fem.Constant(domain,default_scalar_type((0,0)))

# Trial and test functions
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Displacement, velocity, acceleration functions
dn = fem.Function(V)
d = fem.Function(V)
vel_n = fem.Function(V)
vel = fem.Function(V)
an = fem.Function(V)
a = fem.Function(V)
dp = fem.Function(V) # predictor
a.name = "acceleration"
d.name = "displacement"

# Set initial conditions (currently parabola)
c_s = 57 # [m/s]
k_w = 1.875 / L_0
def f_(x):
    c1 = k_w * L_0
    c2 = k_w * x

    return ((np.cos(c1) + np.cosh(c1))*(np.cosh(c2) - np.cos(c2))) + ((np.sin(c1) + np.sinh(c1))*(np.sinh(c2) - np.sin(c2)))

def initial_condition(x):
    c1 = k_w * L_0
    bottom = ((np.cos(c1) + np.cosh(c1))*(np.cosh(c1) - np.cos(c1))) + ((np.sin(c1) + np.sinh(c1))*(np.sinh(c1) - np.sin(c1)))
    v = 0.01 * c_s * (f_(x[0]) / bottom)

    return np.vstack((np.zeros_like(v),v), dtype=default_scalar_type)

vel_n.interpolate(initial_condition)
vel_n.x.scatter_forward()

# Weak form
def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

def mass(u,v):
    return rho * ufl.dot(u,v) * ufl.dx

def stiffness(u,v):
    return ufl.inner(sigma(u),epsilon(v)) * ufl.dx

# Set up time scheme [s]
dt = 0.001
t = 0
t_f = 1
time = np.linspace(t,t_f,int((t_f-t)/dt)+1)

beta = 0.25
gamma = 0.5

a_form = fem.form(mass(u,v) + beta*dt**2 * stiffness(u,v))
L_form = fem.form(ufl.dot(f,v) * ufl.dx + ufl.dot(T,v) * ufl.ds - stiffness(dp,v))

A = assemble_matrix(a_form,bcs=[bcs])
A.assemble()
b = assemble_vector(L_form)

solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
pc = solver.getPC()
pc.setType(PETSc.PC.Type.LU)

vtk = io.VTKFile(domain.comm, "results2/elastodynamics.pvd", "w")

d_tip = np.zeros(len(time))
v_tip = np.zeros(len(time))
ke = np.zeros(len(time))
pe = np.zeros(len(time))

# Energy forms
ke_form = fem.form(0.5 * mass(vel,vel))
pe_form = fem.form(0.5 * stiffness(d,d))

for t in range(0,len(time)):
    vtk.write_function(d, t)

    dp.x.array[:] = dn.x.array + dt*vel_n.x.array + ((dt**2)/2) * (1-2*beta)*an.x.array
    dp.x.scatter_forward()

    with b.localForm() as loc_b:
        loc_b.set(0)
    assemble_vector(b,L_form)

    apply_lifting(b,[a_form],[[bcs]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, [bcs])

    solver.solve(b,a.x.petsc_vec)
    a.x.scatter_forward()

    d.x.array[:] = dn.x.array + dt*vel_n.x.array + ((dt**2)/2) * ((1-2*beta)*an.x.array + 2*beta*a.x.array)
    vel.x.array[:] = vel_n.x.array + dt * ((1-gamma)*an.x.array + gamma*a.x.array)
    d.x.scatter_forward()
    vel.x.scatter_forward()

    ke[t] = fem.assemble_scalar(ke_form)
    pe[t] = fem.assemble_scalar(pe_form)
    
    tip_d = d.eval(points_on_proc, cells)
    tip_v = vel.eval(points_on_proc, cells)
    d_tip[t] = tip_d[1] # vertical displacement
    v_tip[t] = tip_v[1] # vertical velocity

    an.x.array[:] = a.x.array
    dn.x.array[:] = d.x.array
    vel_n.x.array[:] = vel.x.array

A.destroy()
b.destroy()
solver.destroy()

vtk.close()

# print("Shape of time: " + str(np.size(time)))
# print("Shape of d_tip: " + str(np.size(d_tip)))

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(time,d_tip)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Displacement (m)")
ax.set_title("Tip Displacement at (L_0,H_0) vs. Time")
plt.savefig("tip_displacement")

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(time,v_tip)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Velocity (m/s)")
ax.set_title("Tip Velocity at (L_0,H_0) vs. Time")
plt.savefig("tip_velocity")

fig, ax = plt.subplots(figsize=(10,10))
ax.plot(time,ke,c='k',label="KE")
ax.plot(time,pe,c='r',label="PE")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Energy (J)")
ax.set_title("Energy vs. Time")
ax.legend()
plt.savefig("energy")