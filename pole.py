import numpy as n
import math as m
import matplotlib.pyplot as plt

class Node(object):
    def __init__(self,position,transform,ID):
        self.position = position
        self.transform = transform
        self.ID = ID


class Pole2D(object):
    def __init__(self,node,A,E):
        self.node = node
        self.A = A
        self.E = E

    def KE(self):
        n1 = self.node[0]
        n2 = self.node[1]
        x1 = n1.position[0]
        x2 = n2.position[0]
        y1 = n1.position[1]
        y2 = n2.position[1]
        l = m.sqrt((x2-x1)**2+(y2-y1)**2)
        alpha = (x2-x1)/l
        beta = (y2-y1)/l
        alpha2 = alpha**2
        ab = alpha*beta
        beta2 = beta**2
        k = n.matrix([[alpha2,ab,-alpha2,-ab],[ab,beta2,-ab,-beta2],[-alpha2,-ab,alpha2,ab],[-ab,-beta2,ab,beta2]])
        ke = (self.A*self.E/l)*k
        return ke

    def ID(self):
        n1id = self.node[0].ID
        n2id = self.node[1].ID
        return [n1id,n2id]

class System(object):
    def __init__(self,nodes,poles):
        self.nodes = nodes
        self.poles = poles
    def KE(self):
        df = 2*len(self.nodes)
        k = n.zeros([df,df])
        for pole in self.poles:
            poleId = pole.ID()
            i = poleId[0]
            j = poleId[1]
            ke = pole.KE()
            k[2*i-2:2*i,2*i-2:2*i] = k[2*i-2:2*i,2*i-2:2*i] + ke[0:2,0:2]
            k[2*i-2:2*i,2*j-2:2*j] = k[2*i-2:2*i,2*j-2:2*j] + ke[0:2,2:]
            k[2*j-2:2*j,2*i-2:2*i] = k[2*j-2:2*j,2*i-2:2*i] + ke[2:,0:2]
            k[2*j-2:2*j,2*j-2:2*j] = k[2*j-2:2*j,2*j-2:2*j] + ke[2:,2:]
        return k
    def display(self):
        pass
    def solve(self):
        pass



n1 = Node([0,0],[0,0],1)
n2 = Node([0,1],[0,0],2)
n3 = Node([1,1],[0,0],3)
n4 = Node([1,0],[0,0],4)
pole1 = Pole2D([n1,n2],1,1)
pole2 = Pole2D([n2,n3],1,1)
pole3 = Pole2D([n3,n1],1,1)
pole4 = Pole2D([n3,n4],1,1)
pole5 = Pole2D([n2,n4],1,1)
pole6 = Pole2D([n1,n4],1,1)
s = System([n1,n2,n3,n4],[pole1,pole2,pole3,pole4,pole5,pole6])

