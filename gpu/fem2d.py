import cupy as n
import matplotlib.pyplot as plt
import numpy as np


class Node(object):
    def __init__(self, ID, position, load=None):
        self.position = position
        self.ID = ID
        self.load = load


class element2D(object):
    def __init__(self, nodes, t, v, E):
        self.nodes = nodes
        self.v = v
        self.t = t
        self.E = E
        self.t = t

    def KE(self):
        a = abs(self.nodes[1].position[0]-self.nodes[0].position[0])/2
        b = abs(self.nodes[2].position[1]-self.nodes[1].position[1])/2
        al = a/(3*b)
        be = b/(3*a)
        m = (1+self.v)/8
        s = (1-3*self.v)/8
        H = 1/(1-self.v**2)
        r = (1-self.v)/2
        k = [al+r*be, m, r*al/2-be, -s, -be/2-r*al/2, -m, be/2-r*al, s, be +
             r*al, al/2-r*be, al/2-r*be/2, r*be/2-al, -al/2-r*be/2, r*al/2-be]
        ke = n.asarray([[k[8], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                        [k[1], k[0], k[7], k[9], k[5], k[12], k[3], k[11]],
                        [k[2], k[7], k[8], k[5], k[6], k[3], k[4], k[1]],
                        [k[3], k[9], k[5], k[0], k[7], k[11], k[1], k[12]],
                        [k[4], k[5], k[6], k[7], k[8], k[1], k[13], k[3]],
                        [k[5], k[12], k[3], k[11], k[1], k[0], k[7], k[9]],
                        [k[6], k[3], k[4], k[1], k[13], k[7], k[8], k[5]],
                        [k[7], k[11], k[1], k[12], k[3], k[9], k[5], k[0]]])

        return (self.E*H*self.t)*ke

    def ID(self):
        n1id = self.nodes[0].ID
        n2id = self.nodes[1].ID
        n3id = self.nodes[2].ID
        n4id = self.nodes[3].ID
        return [n1id, n2id, n3id, n4id]


class System(object):
    def __init__(self, nodes, element2Ds):
        self.nodes = nodes
        self.el2ds = element2Ds

    def KE(self):
        ke = n.zeros([len(self.nodes)*2, len(self.nodes)*2])
        for el2d in self.el2ds:
            elke = el2d.KE()
            elids = el2d.ID()
            elu = []
            for elid in elids:
                elu.append(2*elid)
                elu.append(2*elid+1)
            for i in range(8):
                for j in range(8):
                    ke[elu[i], elu[j]] = ke[elu[i], elu[j]] + elke[i, j]
        return ke

    def solve(self):
        k1 = self.KE()
        k0 = self.KE()
        P = n.zeros([2*len(self.nodes), 1])
        for node in self.nodes:
            load = node.load
            i = node.ID
            if load is None:
                continue
            tranx = load[0][0]
            trany = load[0][1]
            forcex = load[1][0]
            forcey = load[1][1]
            print(i)
            if trany != None:
                k1[(2*i)+1, (2*i)+1] = k1[(2*i)+1, (2*i)+1]*(10**13)
                P[(2*i)+1] = k1[(2*i)+1, (2*i)+1]*trany
            if tranx != None:
                k1[2*i, 2*i] = k1[2*i, 2*i]*(10**13)
                P[2*i] = k1[2*i, 2*i]*tranx

            P[2*i] = P[2*i]+forcex
            P[2*i+1] = P[2*i+1]+forcey
#        print(k1,P)
        k1np = n.asnumpy(k1)
#        np.savetxt('k1', k1np)
        delta = n.linalg.solve(k1, P)
        Force = n.dot(k0, delta)
        return delta, Force

    def dc(self, h, dens):
        dcs = []
        delta, force = self.solve()
        for i, el2d in enumerate(self.el2ds):
            eldelta = []
            for elid in el2d.ID():
                eldelta.append(delta[2*elid])
                eldelta.append(delta[2*elid+1])
            eldelta = n.asarray(eldelta)
            dc = h*(dens[i])**(h-1)*n.dot(eldelta.T, n.dot(el2d.KE(), eldelta))
            dcs.append(dc[0][0])
        return dcs
    
    def cmat(self,h,dens):
        cs = []
        delta, force = self.solve()
        for i, el2d in enumerate(self.el2ds):
            eldelta = []
            for elid in el2d.ID():
                eldelta.append(delta[2*elid])
                eldelta.append(delta[2*elid+1])
            eldelta = n.asarray(eldelta)
            c = (dens[i])**h*n.dot(eldelta.T, n.dot(el2d.KE(), eldelta))
            cs.append(c[0][0])
        return cs

    def load(self, loads):
        for load in loads:
            self.nodes[load['nodeID']].load = load['load']


class Mesh(object):
    def __init__(self, nx, ny, a, b):
        self.nx = nx
        self.ny = ny
        self.a = a
        self.b = b

    def create(self, E):
        nodes = []
        element2Ds = []
        k = 0
        for i in range(self.nx):
            positionx = i*2*self.a
            for j in range(self.ny):
                positiony = j*2*self.b
                nodes.append(Node(k, [positionx, positiony]))
                k = k+1
        #nodes[len(nodes)-self.ny//2].load = [[None,None],[0,1]]
        for i in range(self.nx-1):
            for j in range(self.ny-1):
                n1 = nodes[i*self.ny+j]
                n4 = nodes[i*self.ny+1+j]
                n2 = nodes[i*self.ny+j+self.ny]
                n3 = nodes[i*self.ny+j+self.ny+1]
                # print(i*(ny-1)+j)
                e = E[i*(self.ny-1)+j]
                # print(e)

                element2Ds.append(element2D([n1, n2, n3, n4], 0.1, 0.2, e))
        return nodes, element2Ds

    def changeE(self, element2Ds, E):
        for n, el2d in enumerate(element2Ds):
            el2d.E = E[n]
        return element2Ds


# print(element2Ds[1].ID())
#nodes[len(nodes)-1].load = [[None,None],[0,1]]
#nodes[0].load = [[0,0],[0,0]]

if __name__ == '__main__':

    # Disable memory pool for device memory (GPU)
    n.cuda.set_allocator(None)

    # Disable memory pool for pinned memory (CPU).
    n.cuda.set_pinned_memory_allocator(None)
# print(element2Ds[1].ID())
    nx = 20
    ny = 40
    elenum = (nx-1)*(ny-1)
    m = Mesh(nx, ny, 1, 1)
    E = 0.5**3*n.ones(elenum)
    nodes, element2Ds = m.create(E)
    s = System(nodes, element2Ds)
    loads1 = [{'nodeID': ny-1, 'load': [[0, 0], [-1, 0]]}, {'nodeID': 0, 'load': [[None, None], [-1, 0]]},
              {'nodeID': nx*ny-ny//2, 'load': [[None, None], [2, 0]]}]
    loads2 = [{'nodeID': nx*ny-ny//3-1, 'load': [[0, 0], [0, 0]]}]
    for i in range(nx-1):
        loads2.append({'nodeID': i, 'load': [[None, None], [-2, 0]]})
    s.load(loads1)
    elenum = (nx-1)*(ny-1)
    dens = n.full(elenum, 0.5)
    print(s.cmat(3,dens))
    delta, force = s.solve()
    deltay = n.abs(delta[1::2])
    deltayimage = deltay.reshape((nx, ny)).T
    deltayimage = n.asnumpy(deltayimage)
    plt.imshow(deltayimage)
    plt.savefig("tmp\\deltay.png")

    forcey = n.abs(force[1::2])
    forceyimage = forcey.reshape((nx, ny)).T
    forceyimage = n.asnumpy(forceyimage)
    plt.imshow(forceyimage)
    plt.savefig("tmp\\forcey.png")

    deltax = n.abs(delta[0::2])
    deltaximage = deltax.reshape((nx, ny)).T
    deltaximage = n.asnumpy(deltaximage)
    plt.imshow(deltaximage)
    plt.savefig("tmp\\deltax.png")

    forcex = n.abs(force[0::2])
    forceximage = forcex.reshape((nx, ny)).T
    forceximage = n.asnumpy(forceximage)
    plt.imshow(forceximage)
    plt.savefig("tmp\\forcex.png")

    np.savetxt('deltax', deltaximage)
    np.savetxt('deltay', deltayimage)
    np.savetxt('forcex', forceximage)
    np.savetxt('forcey', forceyimage)
