from fem2d import *
import numpy as n

def cmat(delta,ke):
    return n.dot(n.dot(delta.T,ke),delta)



nx = 40
ny = 40

m = Mesh(nx,ny)
E = 0.5*n.ones(elenum)
    
nodes,element2Ds = m.create(E)

s = System(nodes,element2Ds)

