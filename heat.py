from firedrake import *

n = 3
mesh = UnitDiskMesh(refinement_level=3)
V= FunctionSpace(mesh, "CG", 1)
x, y = SpatialCoordinate(mesh)
un = Function(V)
R = Constant(0.2)
un.interpolate(exp(-(x**2 + y**2)/R**2))
un1 = Function(V)
Dt = 0.01
dt = Constant(Dt)
v = TestFunction(V)

uh = Constant(0.5)*(un + un1)
a = (un1 - un)*v*dx + dt*inner(grad(uh), grad(v))*dx

aprob = NonlinearVariationalProblem(a, un1)
asolver = NonlinearVariationalSolver(aprob)

t = 0.
tmax = 1.
file0 = File('heat.pvd')
while t < tmax - Dt/2:
    t += Dt

    asolver.solve()
    un.assign(un1)
    file0.write(un)

