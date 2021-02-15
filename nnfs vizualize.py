#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 18:58:07 2021

@author: anonisouris
"""

import pickle
import pygame
from nnfs2 import *
import numpy as np
from time import *
import math
from random import *

def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except:
        return 1
pygame.init()
pygame.font.init()

path=input("Neural Net to vizualize path :\n")
with open(path, 'rb') as config_dictionary_file:
    nn=(pickle.load(config_dictionary_file))

screen = pygame.display.set_mode((1080, 720))

def render_background():
    background=pygame.image.load("zone.png")
    screen.blit(background, (0,0))
render_background()
#Init neurons
class neuron():
    def __init__(self,ntype,index,sx,sy,color):
        self.ntype,self.index,self.sx,self.sy,self.color=ntype,index,sx,sy,color
        self.value=0
    def update(self,nnobj):
        if self.ntype=="i":
            self.value=nnobj.inputs[self.index]
        if self.ntype=="o":
            self.value=nnobj.outputs[self.index]
        if self.ntype=="h":
            self.value=nnobj.hidden[self.index[0],self.index[1]]
            #self.color=(int(255*sigmoid(self.value)),int(255*sigmoid(self.value)),int(255*sigmoid(self.value)))
    def draw_neuron(self):
        if self.ntype=="h":
            self.color=[int(255*sigmoid(self.value))]*3
        if self.ntype=="o":
            self.color=[int(255*(1-self.value)),int(255*self.value),0]
        pygame.draw.circle(screen, self.color, (self.sx,self.sy), 15)
        textsize=int(35 - (len(str(self.value))-1)*1.2)
        myfont = pygame.font.SysFont('Comic Sans MS', textsize)
        textsurface = myfont.render(str(round(self.value,1)), False, (255,0,255))
        screen.blit(textsurface,(self.sx-12,self.sy-7))
        
inputn=list()
between=int(600/nn.i)
y=int(360 - (nn.i*between)/2)
for i in range(nn.i):
    inputn.append(neuron("i",i,50,y,(255,0,0)))
    y+=between
outputn=list()
between=int(600/nn.o)
y=int(360 - (nn.o*between)/2)
for i in range(nn.o):
    outputn.append(neuron("o",i,1030,y,(0,255,0)))
    y+=between
hiddenn=list()
xbetween=int(700/nn.w)
x=int(540 - (nn.w*xbetween)/2)
for xboi in range(nn.w):
    ybetween=int(600/nn.h)
    y=int(360 - (nn.h*ybetween)/2)
    for yboi in range(nn.h):
        hiddenn.append(neuron("h",(xboi,yboi),x,y,(0,0,0)))
        y+=ybetween
    x+=xbetween
all_neurons=inputn+hiddenn+outputn
def draw_neurons():
    global nn
    for i in inputn:
        i.update(nn)
        i.draw_neuron()
    for i in outputn:
        i.update(nn)
        i.draw_neuron()
    for i in hiddenn:
        i.update(nn)
        i.draw_neuron()
def draw_synapses():
    global nn
    overallmax=max([np.max(nn.flw),np.max(nn.whidden)])
    #First layer
    for i in range(np.shape(nn.flw)[0]):
        for j in range(np.shape(nn.flw)[1]):
            if abs(nn.flw[i,j])==nn.flw[i,j]:ncol=(0,0,255)
            else:ncol=(255,165,0)
            pygame.draw.line(screen, ncol, 
                             (inputn[j].sx,inputn[j].sy), (hiddenn[i].sx,hiddenn[i].sy),
                             int(abs(nn.flw[i,j])*10/overallmax)+1)
    #Mid layers
    for k in range(np.shape(nn.whidden)[0]):
        for j in range(np.shape(nn.whidden)[1]):
            for i in range(np.shape(nn.whidden)[2]):
                if abs(nn.whidden[k,j,i])==nn.whidden[k,j,i]:ncol=(0,0,255)
                else:ncol=(255,165,0)
                pygame.draw.line(screen,ncol,
                                 (hiddenn[i+k*nn.h].sx,hiddenn[i+k*nn.h].sy),(hiddenn[j + (k+1)*nn.h].sx,hiddenn[j + (k+1)*nn.h].sy),
                                 int(abs(nn.whidden[k,j,i])*10/overallmax)+1)
    #Output layer
    for i in range(np.shape(nn.woutputs)[0]):
        for j in range(np.shape(nn.woutputs)[1]):
            if abs(nn.woutputs[i,j])==nn.woutputs[i,j]:ncol=(0,0,255)
            else:ncol=(255,165,0)
            pygame.draw.line(screen, ncol, 
                             (hiddenn[j + nn.h*(nn.w-1)].sx,hiddenn[j + nn.h*(nn.w-1)].sy), (outputn[i].sx,outputn[i].sy),
                             int(abs(nn.woutputs[i,j])*10/overallmax)+1)
def nns_update_feed():
    baab=input("Feed me ! ("+str(nn.i)+")\n")
    if baab=="exit":
        pygame.quit()
        exit()
    if baab=="random":
        ine=list()
        for i in range(nn.i):
            ine.append(random()*10)
    else:
        ine=baab.split(",")
    for i in range(len(ine)):ine[i]=float(ine[i])
    nn.feed(ine)


while True:
    try:
        with open(path, 'rb') as config_dictionary_file:
            nn=(pickle.load(config_dictionary_file))
    except:
        print("error")
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    nns_update_feed()
            if event.type == pygame.MOUSEBUTTONDOWN:
                cx,cy = pygame.mouse.get_pos()
                for n in all_neurons:
                    if abs(n.sx-cx)<15 and abs(n.sy-cy)<15:
                        process=1
                        while process:
                            for event in pygame.event.get():
                                if event.type == pygame.QUIT:
                                    pygame.quit()
                                    exit()
                                if event.type == pygame.MOUSEBUTTONUP:
                                    process=0
                            pos = pygame.mouse.get_pos()
                            n.sx,n.sy=pos[0],pos[1]
                            render_background()
                            draw_synapses()
                            draw_neurons()
                            pygame.display.flip()
    try:    
        render_background()                            
        draw_synapses()
        draw_neurons()
    except:
        print("process error")
    pygame.display.flip()