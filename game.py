import pygame
from nnfs2 import *
from math import *
from random import *
import numpy as np
import time
import copy
import pickle
import os
screen = pygame.display.set_mode((720, 480))
clock = pygame.time.Clock()
FPS =60
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
rect = pygame.Rect((0, 0), (32, 32)) 
image = pygame.Surface((32, 32))
image.fill(WHITE)
clock.tick(FPS)
	
track = pygame.image.load("tracke.png")

def draw_terrain():
    screen.blit(track, (0,0))
    """
    for w in waypoints:
        pygame.draw.circle(screen,(0,0,255),w,15)
    screen.fill([47,255,0])
    pygame.draw.rect(screen,BLACK,(100,50,550,50))
    pygame.draw.rect(screen,BLACK,(100,400,550,50))
    pygame.draw.rect(screen,BLACK,(100,50,50,380))
    pygame.draw.rect(screen,BLACK,(600,50,50,380))
    """
waypoints=[(590, 141), (420, 215), (400, 300), (572, 331), (570, 433), (362, 440),
           (298, 337), (247, 253), (101, 247), (52, 168), (83, 87), (188, 76),
           (301, 70), (405, 70)]
# Les waypoints sont des bornes a franchir pour que l'IA gagne des points



class car():  #Classe pour les voitures
    def __init__(self,x,y,keepcolor,col=(255,0,0)):
        self.x=x
        self.y=y
        self.lcheckx=x
        self.lchecky=y
        if keepcolor==0:   # Pour faire de la couleur un attribut génétique
            self.color=sample(range(100, 245), 3)
        self.velocity=0
        self.direction=0
        self.lastmon=time.monotonic()
        self.vision=list()
        self.score=0
        self.ded=0
        self.dedtime=0
        self.nxtwyid=0
    def display(self):
        pygame.draw.polygon(screen, self.color, [(self.x+cos(radians(self.direction))*10,self.y+sin(radians(self.direction))*10),
                                                 (self.x+cos(radians(self.direction+120))*5,self.y+sin(radians(self.direction+120))*5),
                                                 (self.x+cos(radians(self.direction+240))*5,self.y+sin(radians(self.direction+240))*5)])
    def update(self):
       global waypoints
       self.x+=cos(radians(self.direction))*(time.monotonic()-self.lastmon)*self.velocity
       self.y+=sin(radians(self.direction))*(time.monotonic()-self.lastmon)*self.velocity
       self.velocity/=1.1
       self.lastmon=time.monotonic()
       '''
       if sqrt((self.x-self.lcheckx)**2 +(self.y-self.lchecky)**2)>40:
           self.score+=1
           self.lcheckx=self.x
           self.lchecky=self.y
           '''
       if abs(self.x-waypoints[self.nxtwyid][0])<25 and abs(self.y-waypoints[self.nxtwyid][1])<25: # Si l'ia passe un waypoint , elle gagne 10 points
           self.score+=10
           self.nxtwyid+=1
           if self.nxtwyid==len(waypoints):self.nxtwyid=0
    def see(self,showray=0): #input pour les IA (9 distances)
        self.vision=[]
        fov=[-120,-90,-60,-30,0,30,60,90,120]
        for i in fov:
            for j in range(1,700,2):
                try:
                    col=screen.get_at((int(self.x+cos(radians(self.direction+i))*j),int(self.y+sin(radians(self.direction+i))*j)))
                except:
                    break
                if col==(47, 255, 0, 255):
                    break
            if j<=2:
                self.ded = 1
                self.dedtime=time.monotonic()
            if showray==1:
                pygame.draw.rect(screen,self.color,(int(self.x+cos(radians(self.direction+i))*j),int(self.y+sin(radians(self.direction+i))*j),5,5))
            self.vision.append(j) 
        
        
pygame.init()
pop=60 # Nombre d'agents dans la simulation
deds=0
nb=0
dend=0
disp=1
fps=0
brains=list()
bagnoles=list()
for i in range(pop):  # Génération de la population
    brains.append(nnfs(11,randint(1,10),randint(1,10),3)) #Creation de la population
    bagnoles.append(car(500,90,0))
bagnol=car(500,90,0,(255,255,0))  # Voiture à part, controlée par l'homme 
brains[0]=load_nn("voiture-239.37480256967464.brain")   # Importation d'un réseau de neurones déja entrainé
bagnoles[0].color=(255,0,0)
bestscoreofalltime=-10
myfont = pygame.font.SysFont('Comic Sans MS', 50)
while dend==0:
    t0=time.monotonic()
    end=0
    deds=0
    print(nb)
    while deds!=pop and end==0 and time.monotonic()<(t0+25):
        tfps=time.monotonic()
        draw_terrain()
        for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE: #Appuyer sur espace passe a la génération suivante 
                        end=1
                    if event.key == pygame.K_DOWN: #Arrête tout
                        dend=1
                    if event.key == pygame.K_UP:  #Peermet de ce focaliser sur la meilleure voiture
                        if disp==0:disp=1
                        else:disp=0
                    if event.key == pygame.K_z:  # Controles de l'humain sur la voiture jaune
                        bagnol.velocity+=50
                    if event.key == pygame.K_q:
                        bagnol.direction-=20
                    if event.key == pygame.K_d:
                        bagnol.direction+=20


        bagnol.update()  
        bagnol.see()
        bagnol.display()

        #Partie IA
        for i in range(len(bagnoles)): #On itère parmis tous les agents
            if bagnoles[i].ded==1 : #On passe les ia mortes
                continue
            if disp==0 and i==0: #Quand on ce focalise sur une voiture,
                bagnoles[i].see(1) #Permet de voir ce que "voit" l'agent
            else:
                bagnoles[i].see()
            a=brains[i].feed(bagnoles[i].vision+[bagnoles[i].velocity,bagnoles[i].direction])  #On donne a l'agent sa "vision", sa voitesse et direction
            # on stocke la réponse de l'agent dans la variable a
            bagnoles[i].direction+=a[0]*360*(time.monotonic()-bagnoles[i].lastmon)
            bagnoles[i].direction-=1-(a[0]*180*(time.monotonic()-bagnoles[i].lastmon))
            bagnoles[i].velocity=a[1]*150
            bagnoles[i].update()
        for i in range(len(bagnoles)): 
            if bagnoles[i].ded==0 : # On affiche les voitures apres pour eviter que ça gène leur vue
                if disp==0:
                    if i==0:
                        bagnoles[i].display()
                        brains[i].save("stream") # Si on se focalise sur une seule voiture, on sauvegarde l'etat de son "cerveau" dans le fichier stream.brain , pour pouvoir visualiser son activité en diirect, grace au programme "nnfs vizualiser" (de moi)
                else:bagnoles[i].display()
        deds=0
        for i in bagnoles:deds+=i.ded # Comptage du nombre de morts
        textsurface = myfont.render(str(int(fps))+"fps", False, (255,0,0))
        screen.blit(textsurface,(0,0))
        pygame.display.flip()
        fps=1/(time.monotonic()-tfps)
    #Fin du round
    scores=list()
    for i in bagnoles:
        b=0  #Ici, on calcule tous les scores des ia, en fonction de leur distance et du temps
        if i.ded==1:b=-2
        i.score+=b
        if i.dedtime!=0:scores.append(i.score-((i.dedtime-t0)/40))
        else:scores.append(i.score-((time.monotonic()-t0)/40))
        i.score=0
        i.__init__(500,90,1) #On réinitialis toutes les voitures
    print(max(scores))  
    if max(scores)>bestscoreofalltime: #Sauvegarde du meilleur agent dans un fichier
        bigbrain=copy.deepcopy(brains[scores.index(max(scores))])
        try:
            bigbrain.save("voiture-"+str(max(scores)))
            os.remove("voiture-"+str(bestscoreofalltime)+".brain")
        except:
            print("Save error")
        bestscoreofalltime=max(scores)
        print("New highscore !")
    #La partie selection naturelle
    newgen=list()
    best=scores.index(max(scores))
    newgen.append(copy.deepcopy(brains[best]))
    for i in range(8):newgen.append(brains[best].mutate(randint(2,20)))
    scores[best]=-1
    best=scores.index(max(scores))
    newgen.append(copy.deepcopy(brains[best]))
    for i in range(5):newgen.append(brains[best].mutate(randint(2,20)))
    scores[best]=-1
    best=scores.index(max(scores))
    newgen.append(copy.deepcopy(brains[best]))
    for i in range(4):newgen.append(brains[best].mutate(randint(2,20)))
    for i in range(3):newgen.append(copy.deepcopy(choice(brains).mutate(randint(2,20))))
    while len(newgen)<=pop:newgen.append(nnfs(11,randint(1,10),randint(1,10),3))
    brains=newgen
    nb+=1

pygame.quit()

