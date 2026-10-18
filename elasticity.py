from dolfinx import mesh, fem, plot, io, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import ufl
import numpy as np

L = 1.0
W = 0.2
mu = 1.0
rho = 1.0
delta = W / L
gamma = 0.4 * delta**2
beta = 1.25
lambda_ = beta
g = gamma

domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0,0,0]), np.array([L,W,W])],
    [20,6,6],
    cell_type=mesh.CellType.hexahedron
)
V = fem.functionspace(domain, ("Lagrange",1,(domain.geometry.dim,)))

def clamped_boundary(x):
    return np.isclose(x[0],0)

fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain,fdim,clamped_boundary)
u_D = np.array([0,0,0], dtype=default_scalar_type)
dofs = fem.locate_dofs_topological(V,fdim,boundary_facets)
bc = fem.dirichletbc(u_D,dofs,V)

T = fem.Constant(domain,default_scalar_type((0,0,0)))
ds = ufl.Measure("ds",domain=domain)

def epsilon(u):
    return ufl.sym(ufl.grad(u))

def sigma(u):
    return lambda_ * ufl.nabla_div(u) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
f = fem.Constant(domain,default_scalar_type((0,0,-rho*g)))
a = ufl.inner(sigma(u),epsilon(v)) * ufl.dx
L = ufl.dot(f,v) * ufl.dx + ufl.dot(T,v) * ds

problem = LinearProblem(
        a,
        L,
        bcs=[bc],
        petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        petsc_options_prefix="linear_elasticity"
)

uh = problem.solve()

with io.XDMFFile(domain.comm,"deformation.xdmf","w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "Deformation"
    xdmf.write_function(uh)

s = sigma(uh) - 1.0 / 3 * ufl.tr(sigma(uh)) * ufl.Identity(len(uh))
von_Mises = ufl.sqrt(3.0 / 2 * ufl.inner(s, s))

V_von_mises = fem.functionspace(domain, ("DG", 0))
stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points)
stresses = fem.Function(V_von_mises)
stresses.interpolate(stress_expr)

with io.XDMFFile(domain.comm,"stresses.xdmf","w") as xdmf:
    xdmf.write_mesh(domain)
    stresses.name = "Stresses"
    xdmf.write_function(stresses)