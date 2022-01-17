from firedrake import *

import argparse
parser = argparse.ArgumentParser(description='Poisson demo for FE Course.')
parser.add_argument('--n', type=int, default=40, help='Number of rows in grid (and columns).')
parser.add_argument('--problem', type=int, default=1, help='Problem number (1-4).')
args = parser.parse_known_args()
args = args[0]
n = args.n
problem = args.problem

mesh = UnitSquareMesh(n, n)
V = FunctionSpace(mesh, "CG", 1)
u = TrialFunction(V)
v = TestFunction(V)
x, y = SpatialCoordinate(mesh)

if problem == 1:
    a = inner(grad(v), grad(u))*dx
    L = 2*pi**2*sin(pi*x)*sin(pi*y)*v*dx
    bcs = [DirichletBC(V, 0., 1),
           DirichletBC(V, 0., 2)]
    name = 'vanilla'
elif problem == 2:
    a = inner(grad(v), (2 + sin(2*pi*x))*grad(u))*dx
    L = exp(cos(2*pi*x))*v*dx
    bcs = [DirichletBC(V, 0., 1),
           DirichletBC(V, 0., 2),
           DirichletBC(V, 0., 3),
           DirichletBC(V, 0., 4)]
    name = 'coeff'
elif problem == 3:
    a = inner(grad(v), grad(u))*dx
    L = exp(x*y)*v*dx
    bcs = [DirichletBC(V, x*(1-x), 1),
           DirichletBC(V, x*(1-x), 2),
           DirichletBC(V, x*(1-x), 3),
           DirichletBC(V, x*(1-x), 4)]
    name = 'inhomBCs'
elif problem == 4:
    # v*du/dn*dS = v*(u - x(1-x))*dS
    a = inner(grad(v), grad(u))*dx + u*v*ds
    L = (1/(1+x**2 + y**2))*v*dx + x*(1-x)*v*ds
    bcs = []
    name = 'robin'

u0 = Function(V)
prob = LinearVariationalProblem(a, L, u0, bcs=bcs)
solver = LinearVariationalSolver(prob)

solver.solve()
file0 = File(name+'_'+str(n)+'.pvd')
file0.write(u0)
