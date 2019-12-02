import numpy as n

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



nx = 12
ny = 8
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
        n1 = nodes[i*8+j]
        n4 = nodes[i*8+1+j]
        n2 = nodes[i*8+j+8]
        n3 = nodes[i*8+j+9]
        element2Ds.append(element2D([n1,n2,n3,n4],1,0.5,1))
#print(element2Ds[1].ID())
nodes[len(nodes)-1].load = [[None,None],[0,10]]
nodes[0].load = [[0,0],[0,0]]


#print(element2Ds[1].ID())
s = System(nodes,element2Ds)
delta,force = s.solve()
print(force[:16])