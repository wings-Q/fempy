from fem2d import *
import cupy as cp
import math

# Disable memory pool for device memory (GPU)
cp.cuda.set_allocator(None)

# Disable memory pool for pinned memory (CPU).
cp.cuda.set_pinned_memory_allocator(None)


nx = 25
ny = 50
den = 0.5
h = 3
m = Mesh(nx, ny, 1, 1)
loads1 = [{'nodeID': ny-1, 'load': [[0, 0], [0, 0]]},
          {'nodeID': 0, 'load': [[0, 0], [0, 0]]},
          {'nodeID': nx*ny-ny//2, 'load': [[None, None], [2, 0]]}]
loads2 = [{'nodeID': nx-1, 'load': [[None, 0], [0, 0]]},
          {'nodeID': nx*ny-1, 'load': [[None, 0], [0, 0]]},
          {'nodeID': ny*(nx//3*2)-1, 'load': [[None, None], [0, 2]]}]
elenum = (nx-1)*(ny-1)
dens = cp.full(elenum, den)
E = dens**h
nodes, element2Ds = m.create(E)


def OC(nx, ny, dens, den, dc):
    l1 = 0
    l2 = 100000
    move = 0.2
    while(l2-l1 > 10**(-4)):
        lmid = 0.5*(l2+l1)
        densnew = []
        for n, densipy in enumerate(dens):
            # print(dc[n])
            dennew = max((0.001, max(
                densipy-move, min(1.0, min(densipy+move, densipy*math.sqrt(dc[n]/lmid))))))
            if isinstance(dennew, float):
                pass
            else:
                dennew = dennew.tolist()
            # print(type(dennew))
            densnew.append(dennew)
        if sum(densnew) - den*nx*ny > 0:
            l1 = lmid
        else:
            l2 = lmid
    # print(densnew)
    return cp.asarray(densnew)


def dc(system, w, dens):
    dcsharp = system[0].dc(h, dens)
    dc = cp.zeros_like(dcsharp)
    for n, s in enumerate(system):
        dcmat = ((w[n]**h)*h*((s.cmat(h, dens)-cp.min(s.cmat(h, dens))) /
                              (cp.max(s.cmat(h, dens))-cp.min(s.cmat(h, dens))))**(h-1))*s.dc(h, dens)
        dc = dc + dcmat
    return dc**(1/h)


for i in range(30):
    # print(dens)
    E = dens**h
    element2Ds = m.changeE(element2Ds, E)
    s1 = System(nodes, element2Ds)
    s2 = System(nodes, element2Ds)
    s1.load(loads1)
    s2.load(loads2)
    w = [0.5, 0.5]
    dc = s1.dc(h, dens)
#    print(dc)
    densnew = OC(nx, ny, dens, den, dc)
    densimage = cp.asarray(densnew)
    denimage = densimage.reshape((nx-1, ny-1)).T
    denimage = cp.asnumpy(denimage)
    plt.imshow(denimage)
    filename = "gpu\\tmp\\denimage"+str(i+1)+".png"
    plt.savefig(filename)
    if cp.max(cp.abs(dens-densnew)).tolist() < 0.001:
        break
    dens = densnew
