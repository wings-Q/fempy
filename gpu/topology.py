from fem2d import *
import cupy as cp
from sko.GA import GA
from sko.PSO import PSO
import math

nx = 60
ny = 60
den = 0.5
h = 3
m = Mesh(nx,ny)
elenum = (nx-1)*(ny-1)
dens = cp.full(elenum,den)

def OC(nx,ny,dens,den,dc):
    l1 = 0
    l2 = 100000
    move = 0.2
    while(l2-l1 > 10**(-4)):
        lmid = 0.5*(l2+l1)
        densnew = []
        for n,densipy in enumerate(dens):
            #print(dc[n])
            dennew = max((0.001,max(densipy-move,min(1.0,min(densipy+move,densipy*math.sqrt(-dc[n]/lmid))))))            
            if isinstance(dennew, float):
                pass
            else:
                dennew = dennew.tolist()
            #print(type(dennew))
            densnew.append(dennew)
        if sum(densnew) - den*nx*ny > 0:
            l1 = lmid
        else:
            l2 = lmid
    #print(densnew)
    return cp.asarray(densnew)

for i in range(30):
    #print(dens)
    E = dens**h
    nodes,element2Ds = m.create(E)
    s = System(nodes,element2Ds)
    dc = s.dc(h,dens)
    densnew = OC(nx,ny,dens,den,dc)
    densimage = cp.asarray(densnew)
    denimage = densimage.reshape((nx-1,ny-1)).T
    denimage = cp.asnumpy(denimage)
    plt.imshow(denimage)
    filename = "gpu\\tmp\\denimage"+str(i+1)+".png"
    plt.savefig(filename)
    if cp.max(cp.abs(dens-densnew)).tolist() < 0.001:
        break
    dens = densnew


