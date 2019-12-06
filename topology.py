from fem2d import *
import numpy as n

def cmat(delta,ke):
    return n.dot(n.dot(delta.T,ke),delta)



nx = 40
ny = 40

m = Mesh(nx,ny)
nodes,element2Ds = m.create()

s = System(nodes,element2Ds)

