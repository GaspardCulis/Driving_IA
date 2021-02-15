# -*- coding: utf-8 -*-

import numpy as np
import copy
from random import *

def af(x): #Fonction d'activation des neurones de sortie, ici la fonction sigmoïde 
    return(1/(1+np.exp(-x)))
def sigmoid(x):return(1/(1+np.exp(-x)))
class nnfs():
    def __init__(self,i,w,h,o): # i:Le nombre d'entrées; w:nombre de hidden layers; h:nombre de neurones par hidden layer; o:le nombre de sorties
        self.i=i
        self.w=w
        self.o=o
        self.h=h
        self.inputs=np.array([0]*i,dtype=float)
        self.hidden=np.full((w,h), 0,dtype=float)
        self.outputs=np.array([0]*o,dtype=float)
        self.flw=np.random.rand(h,i)-0.5 #FLW=FirstLayerWheights
        self.whidden=np.random.rand(w-1,h,h)-0.5
        self.bhidden=np.random.rand(w,h)-0.5
        self.woutputs=np.random.rand(o,h)-0.5
        self.boutputs=np.random.rand(o)-0.5
    def feed(self,inputs): #On donne des valeurs au reseau de neurones (un liste de taille i) et il renvoie une liste de taille o
        self.inputs=np.array(inputs)
        self.hidden[0]=(np.dot(self.flw,self.inputs)+self.bhidden[0])
        for i in range(0,self.w-1):
            self.hidden[i+1]=(np.dot(self.whidden[i],self.hidden[i])+self.bhidden[i+1])
        self.outputs=af(np.dot(self.woutputs,self.hidden[self.w-1])+self.boutputs)
        return(self.outputs)
    def mutate(self,c): #Fonction qui renvoie une version du reseau de neurones legerement modifiée, modifiée par un facteur c
        temp=copy.deepcopy(self)
        randa=np.random.rand(self.w-1,self.h,self.h)
        temp.whidden+=(randa-0.5)/c
        randa=np.random.rand(self.h,self.i)
        temp.flw+=(randa-0.5)/c
        randa=np.random.rand(self.w,self.h)
        temp.bhidden+=(randa-0.5)/c
        randa=np.random.rand(self.o,self.h)
        temp.woutputs+=(randa-0.5)/c
        randa=np.random.rand(self.o)
        temp.boutputs+=(randa-0.5)/c
        return temp
    def save(self,name): # Sauvegarde d'un reseau de neurones dans un fichier.brain
        import pickle
        with open(name+'.brain', 'wb') as config_dictionary_file:
            pickle.dump(self, config_dictionary_file) 


def load_nn(path): # recuperer un reseau de neurones sauvegardé
        import pickle
        with open(path, 'rb') as config_dictionary_file:
            return(pickle.load(config_dictionary_file))