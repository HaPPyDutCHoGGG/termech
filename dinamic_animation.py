import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

t = np.linspace(0,10,10001)
#(m1,m2,a,b,l0,c,g) - start params
#  y[0,1,2,3] = phi,psi,phi',psi'
# dy[0,1,2,3] = phi',psi',phi'',psi''
m1 = 50; m2 = 0.5; a, b, l0 = 1,1,1; c = 250; g = 9.8;
phi = 2*t
psi = 4*(t**2)

x0, y0 = 0, 0
xD, yD = x0, y0
xA, yA = (x0 + a*np.cos(phi)), (y0 + a*np.sin(phi))
xE, yE = (x0 + 2*a*np.cos(phi)), (y0 + 2*a*np.sin(phi))
xB, yB = (x0 + a*np.cos(phi) + b*np.sin(psi)), (y0 +b*np.cos(psi) - a*np.sin(phi))
xC, yC = x0 + 2*a, y0 + l0

fig = plt.figure(figsize=[15,12])
ax = fig.add_subplot(1,1,1)
ax.set(xlim=[0,10],ylim=[-5,5])
ax.axis('equal')

D = ax.plot(xD, yD, 'o', color=[1,0,0])[0]
A = ax.plot(xA[0], yA[0], 'o', color='red')[0]
E = ax.plot(xE[0], yE[0], 'o', color='blue')[0]
B = ax.plot(xB[0], yB[0], 'o', color='green')[0]
C = ax.plot(xC, yC, 'o', color=[1,0,0])[0]

DE = ax.plot([xD,yD], [xE[0],yE[0]], color='red')[0]
AB = ax.plot([xA[0],yA[0]], [xB[0],yB[0]], color='green')[0]

def kadr(i):
    D.set_data(xD,yD)
    A.set_data(xA[i],yA[i])
    E.set_data(xE[i],yE[i])
    B.set_data(xB[i],yB[i])
    C.set_data(xC,yC)
    
    DE.set_data([xD,yD], [xE[i],yE[i]])
    AB.set_data([xA[i],yA[i]], [xB[i],yB[i]])
    
    return [D, A, E, B, C, DE, AB]

kino = FuncAnimation(fig, kadr, interval = t[1]-t[2], frames=len(t))

plt.show()
