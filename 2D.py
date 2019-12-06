import numpy as n
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self,ID,position,load=None):
        self.position = position
        self.ID = ID
        self.load = load

class element2D(object):
    def __init__(self,nodes,t,v,E):
        self.nodes = nodes
        self.v = v
        self.t = t
        self.E = E
        self.t = t

    def KE(self):
        a = (self.nodes[0].position[0]-self.nodes[1].position[0])/2
        b = (self.nodes[2].position[1]-self.nodes[1].position[1])/2
        al = a/(3*b)
        be = b/(3*a)
        m = (1+self.v)/8
        s = (1-3*self.v)/8
        H = 1/(1-self.v**2)
        r = (1-self.v)/2
        k = [al+r*be,m,r*al/2-be,-s,-be/2-r*al/2,-m,be/2-r*al,s]
        ke = n.matrix([[k[0],k[1],k[2],k[3],k[4],k[5],k[6],k[7]],[k[1],k[0],k[7],k[6],k[5],k[4],k[3],k[2]],[k[2],k[7],k[0],k[5],k[6],k[3],k[4],k[1]],[k[3],k[6],k[5],k[0],k[7],k[2],k[1],k[4]],[k[4],k[5],k[6],k[7],k[0],k[1],k[2],k[3]],[k[5],k[4],k[3],k[2],k[1],k[0],k[7],k[6]],[k[6],k[3],k[4],k[1],k[2],k[7],k[0],k[5]],[k[7],k[2],k[1],k[4],k[3],k[6],k[5],k[0]]])
        
        return (self.E*H*self.t)*ke
        
    def ID(self):
        n1id = self.nodes[0].ID
        n2id = self.nodes[1].ID
        n3id = self.nodes[2].ID
        n4id = self.nodes[3].ID
        return [n1id,n2id,n3id,n4id]

class System(object):
    def __init__(self,nodes,element2Ds):
        self.nodes = nodes
        self.el2ds = element2Ds

    def KE(self):
        ke = n.zeros([len(self.nodes)*2,len(self.nodes)*2])
        for el2d in self.el2ds:
            elke = el2d.KE()
            elids = el2d.ID()
            elu = []
            for elid in elids:
                elu.append(2*elid)
                elu.append(2*elid+1)
            for i in range(8):
                for j in range(8):
                    ke[elu[i],elu[j]] = ke[elu[i],elu[j]] + elke[i,j]
        return ke
    def solve(self):
        k1 = self.KE()
        k0 = self.KE()
        P = n.zeros([2*len(self.nodes),1])
        for node in self.nodes:
            load = node.load
            i = node.ID
            if load is None:
                continue
            tranx = load[0][0]
            trany = load[0][1]
            forcex = load[1][0]
            forcey = load[1][1]
            if tranx != None:
                k1[2*i-2,2*i-2] = k1[2*i-2,2*i-2]*(10**8)
                P[2*i-2] = k1[2*i-2,2*i-2]*tranx
            if trany != None:
                k1[2*i-1,2*i-1] = k1[2*i-1,2*i-1]*(10**8)
                P[2*i-1] = k1[2*i-1,2*i-1]*trany
            P[2*i-2] = P[2*i-2]+forcex
            P[2*i-1] = P[2*i-1]+forcey
        delta = n.dot(n.linalg.inv(k1),P)
        Force = n.dot(k0,delta)
        return delta,Force



nx = 120
ny = 80
vol = [12,8]
nodes = []
element2Ds = []
k = 0
for i in range(nx):
    positionx = i
    for j in range(ny):
        positiony = j
        if i == 0:
            nodes.append(Node(k,[i,j],[[0,None],[0,0]]))
        else:
            nodes.append(Node(k,[i,j]))
        k = k+1
for i in range(nx-1):
    for j in range(ny-1):
        n1 = nodes[i*ny+j]
        n4 = nodes[i*ny+1+j]
        n2 = nodes[i*ny+j+ny]
        n3 = nodes[i*ny+j+ny+1]
        element2Ds.append(element2D([n1,n2,n3,n4],1,0.2,100))
#print(element2Ds[1].ID())
nodes[len(nodes)-1].load = [[None,None],[0,1]]
nodes[0].load = [[0,0],[0,0]]


#print(element2Ds[1].ID())
s = System(nodes,element2Ds)
delta,force = s.solve()
deltay = n.abs(delta[1::2])
deltayimage = deltay.reshape((nx,ny)).T
plt.imshow(deltayimage)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.savefig("./tmp/deltay.png")

forcey = n.abs(force[1::2])
forceyimage = forcey.reshape((nx,ny)).T
plt.imshow(forceyimage)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.savefig("./tmp/forcey.png")

deltax = n.abs(delta[0::2])
deltaximage = deltax.reshape((nx,ny)).T
plt.imshow(deltaximage)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.savefig("./tmp/deltax.png")

forcex = n.abs(force[0::2])
forceximage = forcex.reshape((nx,ny)).T
plt.imshow(forceximage)
cax = plt.axes([0.85, 0.1, 0.075, 0.8])
plt.colorbar(cax=cax)
plt.savefig("./tmp/forcex.png")