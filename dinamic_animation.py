import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint


t = np.linspace(0,10,1000)
#(m1,m2,a,b,l0,c,g) - start params
#  y[0,1,2,3] = phi,psi,phi',psi'
# dy[0,1,2,3] = phi',psi',phi'',psi''
m1 = 50; m2 = 0.5; a, b, l0 = 1,1,1; c = 250; g = 9.8;
phi = t
psi = t**2

x0, y0 = 0, 0
xD, yD = x0, y0
xA, yA = (x0 + a*np.cos(phi)), (y0 + a*np.sin(phi))
xE, yE = (x0 + 2*a*np.cos(phi)), (y0 + 2*a*np.sin(phi))
#xB, yB = (x0 + a*np.cos(phi) + b*np.sin(psi)), (a*np.sin(phi) - y0 +b*np.cos(psi))
xB, yB = xA + b*np.sin(psi), yA - b*np.cos(psi)
xC, yC = x0 + 2*a, y0 + l0

#n = 13; h = 0.05; ss = 0                                           #spring 
#xP, yP = np.linspace(0,1,(n)), np.zeros(n)
#Lx, Ly = (2*a*(1 - np.cos(phi))), (l0 - 2*a*np.sin(phi))
#for i in range(1,len(yP)):
#    yP[i] = h * np.sin(ss)
#    ss += np.pi / 2


fig = plt.figure(figsize=[13,9])
ax = fig.add_subplot(1,1,1)
ax.axis('equal')
ax.set(xlim=[-5,5],ylim=[-4,4])

#spring = ax.plot([(xP[0] + xE[0])*Lx], [yP[0]*(Ly + yE[0])], color='green')[0]
spring = ax.plot([xE[0],xC], [yE[0],yC], color='green')[0]  #затычка
n = 10; h = 0.05
L = np.sqrt(8*(a**2)*(1-np.cos(phi)) + l0*(l0 - 4*a*np.sin(psi)))
Lx, Ly = (2*a*(1 - np.cos(phi))), (l0 - 2*a*np.sin(phi))
Ln = L/13
alpha = np.arctan(L/(2*h))
theta = np.arctan(Ly/Lx) 
R = h / np.cos(alpha)
xPs, yPs = Ln*np.cos(theta), Ln*np.sin(theta)

Ps_0 = ax.plot(xE[0] + xPs[0], yE[0] + yPs[0], 'o',color='blue')[0]
Ps_1 = ax.plot(xE[0] + 2*xPs[0], yE[0] + 2*yPs[0], 'o',color='blue')[0]
Ps_2 = ax.plot(xE[0] + 3*xPs[0], yE[0] + 3*yPs[0], 'o',color='blue')[0]
Ps_3 = ax.plot(xE[0] + 4*xPs[0], yE[0] + 4*yPs[0], 'o',color='blue')[0]
Ps_4 = ax.plot(xE[0] + 5*xPs[0], yE[0] + 5*yPs[0], 'o',color='blue')[0]
Ps_5 = ax.plot(xE[0] + 6*xPs[0], yE[0] + 6*yPs[0], 'o',color='blue')[0]
Ps_6 = ax.plot(xE[0] + 7*xPs[0], yE[0] + 7*yPs[0], 'o',color='blue')[0]
Ps_7 = ax.plot(xE[0] + 8*xPs[0], yE[0] + 8*yPs[0], 'o',color='blue')[0]
Ps_8 = ax.plot(xE[0] + 9*xPs[0], yE[0] + 9*yPs[0], 'o',color='blue')[0]
Ps_9 = ax.plot(xE[0] + 10*xPs[0], yE[0] + 10*yPs[0], 'o',color='blue')[0]


wall_vertical = ax.plot([0, 0], [0, 3], color='blue', linewidth = 5)    
wall_horizontal = ax.plot([0, 3], [0, 0], color='blue', linewidth = 5)

DE = ax.plot([xD,xE[0]], [yD,yE[0]], color='red', linewidth = 4)[0]     
AB = ax.plot([xA[0],xB[0]], [yA[0],yB[0]], color='green')[0]


D = ax.plot(xD, yD, 'o', color='red')[0]
A = ax.plot(xA[0], yA[0], 'o', color='red')[0]
E = ax.plot(xE[0], yE[0], 'o', color='red')[0]
B = ax.plot(xB[0], yB[0], 'o', color='green')[0]
C = ax.plot(xC, yC, 'o', color='green')[0]

def kadr(i):
    D.set_data(xD,yD)
    A.set_data(xA[i],yA[i])
    E.set_data(xE[i],yE[i])
    B.set_data(xB[i],yB[i])
    C.set_data(xC,yC)
    
    DE.set_data([xD,xE[i]], [yD,yE[i]])
    AB.set_data([xA[i],xB[i]], [yA[i],yB[i]])
    #spring.set_data([(xP[i] + xE[i])*Lx], [(yP[i]*+ yE[i])*Ly])
    spring.set_data([xE[i],xC], [yE[i],yC])

    Ps_0.set_data(xE[i] + xPs[i], yE[i] + yPs[i])
    Ps_1.set_data(xE[i] + 2*xPs[i], yE[i] + 2*yPs[i])
    Ps_2.set_data(xE[i] + 3*xPs[i], yE[i] + 3*yPs[i])
    Ps_3.set_data(xE[i] + 4*xPs[i], yE[i] + 4*yPs[i])
    Ps_4.set_data(xE[i] + 5*xPs[i], yE[i] + 5*yPs[i])
    Ps_5.set_data(xE[i] + 6*xPs[i], yE[i] + 6*yPs[i])
    Ps_6.set_data(xE[i] + 7*xPs[i], yE[i] + 7*yPs[i])
    Ps_7.set_data(xE[i] + 8*xPs[i], yE[i] + 8*yPs[i])
    Ps_8.set_data(xE[i] + 9*xPs[i], yE[i] + 9*yPs[i])
    Ps_9.set_data(xE[i] + 10*xPs[i], yE[i] + 10*yPs[i])
    

    
    return [D, A, E, B, C, DE, AB, spring, 
            Ps_9,Ps_8,Ps_7,Ps_6,Ps_5,Ps_4,Ps_3,Ps_2,Ps_1,Ps_0]

kino = FuncAnimation(fig, kadr, interval = t[1]-t[2], frames=len(t))

plt.show()
