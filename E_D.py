# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 19:05:47 2023

@author: gabri
"""
import random as rn
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


n_geracoes= 1000
tam_pop= 1000
f_escala= 0.2
prop_mut= 0.3


def Ackley(x, y):
    a = 20
    b = 0.2
    c = 2 * np.pi
    term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2 + y**2)))
    term2 = -np.exp(0.5 * (np.cos(c*x) + np.cos(c*y)))
    return a + np.exp(1) + term1 + term2

def E_D(n_geracoes, tam_pop, f_escala, prop_mut, forward):
    
    # gera população inicial
    gammas= np.random.randint(low=-5 ,high=5 ,size=(tam_pop, 2))
    fitnes_pop= []

    
    # calcula erro da pop inicial
    for i in range(tam_pop):
        a= forward(x=gammas[i,0],y=gammas[i,1] )
        fitnes_pop.append(a)
    
    for i in range(n_geracoes):  
        new_gammas= []
        fitnes_new_popy=[]
        print('geracao==',i)
        for j in range(tam_pop):
    
            #mutacao
            indices= rn.sample(range(0,tam_pop), 3)
            u_g= gammas[indices[0]]+ f_escala*(gammas[indices[1]]-gammas[indices[2]])
    
            #cruzamento
            for k in range(2):
                if rn.random()>prop_mut:                    
                    u_g[k]= gammas[j][k] 
    
            a= forward(x= u_g[0], y= u_g[1] )
            fitnes_u_g= a 
            fitnes_x= fitnes_pop[j]
            #selecao
            
            if fitnes_u_g<fitnes_x:
                
                fitnes_new_popy.append( fitnes_u_g)
                new_gammas.append( u_g)
            else:               
                fitnes_new_popy.append( fitnes_x)
                new_gammas.append( gammas[j])
         
        
    gammas= deepcopy(new_gammas)
    fitnes_pop= deepcopy(fitnes_new_popy)
    menor= float ('inf')
    
    for i in range(tam_pop):        
        if menor > fitnes_pop[i]:
            menor= fitnes_pop[i]
            indice=i
            
    print(gammas[indice])
    
    
if __name__ == '__main__':
    E_D(n_geracoes= 1000, tam_pop=1000, f_escala= 0.2, prop_mut= 0.3, forward= Ackley)
    
    import numpy as np
    

    
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = Ackley(X, Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()