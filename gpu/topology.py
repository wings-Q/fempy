from fem2d import *
import cupy as cp
from sko.GA import GA
from sko.PSO import PSO

nx = 20
ny = 20
den = 0.5
h = 3
m = Mesh(nx,ny)
elenum = (nx-1)*(ny-1)
dens = cp.full(elenum,den)
dens0 = [0 for n in range(elenum)]
dens1 = [1 for n in range(elenum)]
def opfunc(densipy):
    dens = []
    for i in densipy:
        dens.append(i)
    dens = cp.asarray(dens)
    E = dens**h
    nodes,element2Ds = m.create(E)
    s = System(nodes,element2Ds)
    delta,force = s.solve()
    return cp.dot(delta.T,force)

ga = PSO(func=opfunc,dim = elenum,lb=dens0,ub=dens1)

from sko.operators import crossover

#ga.register(operator_name='crossover', operator=crossover.crossover_1point)

for i in range(150):
    denmat = ga.run(10)
    #print(denmat)
    #denlist=[]
    #print(ga.gbest_x)
    #print(ga.gbest_y)
    denimage = ga.gbest_x.reshape((nx-1,ny-1)).T
    plt.imshow(denimage)
    filename = "tmp\\denimage"+str(i+1)+".png"
    plt.savefig(filename)
