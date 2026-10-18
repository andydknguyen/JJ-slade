import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc
from dolfinx.mesh import create_box, CellType

L = 8.0
H = 0.2
B = 0.1
domain = create_box(
    MPI.COMM_WORLD,
    [[0.0, -B / 2, -H / 2], [L, B / 2, H / 2]],
    [8, 2, 2],
    CellType.hexahedron,
)

dim = domain.topology.dim
dx = ufl.Measure("dx", domain=domain)

degree = 2
shape = (dim,)
V = fem.functionspace(domain, ("Q", degree, shape))

u = fem.Function(V, name="Displacement")

def left(x):
    return np.isclose(x[0], 0.0)


def point(x):
    return np.isclose(x[0], L) & np.isclose(x[1], 0) & np.isclose(x[2], 0)


clamped_dofs = fem.locate_dofs_geometrical(V, left)
point_dof = fem.locate_dofs_geometrical(V, point)[0]
point_dofs = np.arange(point_dof * dim, (point_dof + 1) * dim)


bcs = [fem.dirichletbc(np.zeros((dim,)), clamped_dofs, V)]

E = fem.Constant(domain, 210e3)
nu = fem.Constant(domain, 0.3)
rho = fem.Constant(domain, 7.8e-3)
f = fem.Constant(domain, (0.0,) * dim)

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)


def epsilon(v):
    return ufl.sym(ufl.grad(v))


def sigma(v):
    return lmbda * ufl.tr(epsilon(v)) * ufl.Identity(dim) + 2 * mu * epsilon(v)

u_old = fem.Function(V)
v_old = fem.Function(V)
a_old = fem.Function(V)
a_new = fem.Function(V)
v_new = fem.Function(V)

beta_ = 0.25
beta = fem.Constant(domain, beta_)
gamma_ = 0.5
gamma = fem.Constant(domain, gamma_)
dt = fem.Constant(domain, 0.0)

a = 1 / beta / dt**2 * (u - u_old - dt * v_old) + a_old * (1 - 1 / 2 / beta)
a_expr = fem.Expression(a, V.element.interpolation_points)

v = v_old + dt * ((1 - gamma) * a_old + gamma * a)
v_expr = fem.Expression(v, V.element.interpolation_points)

u_ = ufl.TestFunction(V)
du = ufl.TrialFunction(V)

eta_M = fem.Constant(domain, 1e-4)
eta_K = fem.Constant(domain, 1e-4)


def mass(u, u_):
    return rho * ufl.dot(u, u_) * ufl.dx


def stiffness(u, u_):
    return ufl.inner(sigma(u), epsilon(u_)) * ufl.dx


def damping(u, u_):
    return eta_M * mass(u, u_) + eta_K * stiffness(u, u_)


Residual = mass(a, u_) + damping(v, u_) + stiffness(u, u_) - ufl.dot(f, u_) * ufl.dx

Residual_du = ufl.replace(Residual, {u: du})
a_form = ufl.lhs(Residual_du)
L_form = ufl.rhs(Residual_du)

problem = fem.petsc.LinearProblem(
    a_form, L_form, u=u, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="linear_elasticity_"
)

vtk = io.VTKFile(domain.comm, "results/elastodynamics.pvd", "w")

t = 0.0

Nsteps = 400
Nsave = 100
times = np.linspace(0, 2, Nsteps + 1)
save_freq = Nsteps // Nsave

for i, dti in enumerate(np.diff(times)):
    if i % save_freq == 0:
        vtk.write_function(u, t)

    dt.value = dti
    t += dti

    if t <= 0.2:
        f.value = np.array([0.0, 1.0, 1.5]) * t / 0.2
    else:
        f.value *= 0.0

    problem.solve()

    u.x.scatter_forward()  # updates ghost values for parallel computations

    # compute new velocity v_n+1
    v_new.interpolate(v_expr)

    # compute new acceleration a_n+1
    a_new.interpolate(a_expr)

    # update u_n with u_n+1
    u_old.x.array[:] = u.x.array

    # # update v_n with v_n+1
    # v_old.x.array[:] = v.x.array

    # # update a_n with a_n+1
    # a_old.x.array[:] = a.x.array

    clear_output(wait=True)
    print(f"Time increment {i+1}/{Nsteps}")

vtk.close()