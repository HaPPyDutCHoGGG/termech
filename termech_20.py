import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import sympy as sp

t = sp.Symbol('t')
x = sp.sin(t)
y = sp.sin(2*t)
Vx = sp.diff(x,t)
Vy = sp.diff(y,t)
#print(x,y,Vx,Vy)
F_x = sp.lambdify(t, x)
F_y = sp.lambdify(t, y)
F_Vx = sp.lambdify(t, Vx)
F_Vy = sp.lambdify(t, Vy)


t = np.linspace(0,10,1001)

x = F_x(t)
y = F_y(t)
Vx = F_Vx(t)
Vy = F_Vy(t)

fig = plt.figure(figsize=[10,10])
ax = fig.add_subplot(1,1,1)
ax.axis('equal')
ax.set(xlim=[-2,2], ylim=[-2,2])

ax.plot(x,y)
P = ax.plot(x[0],y[0],marker='o')[0]
V_line = ax.plot([x[0], x[0]+Vx[0]], [y[0], y[0]+Vy[0]])[0]

def TMoM(i):
    P.set_data(x[i],y[i])
    V_line.set_data([x[i], x[i]+Vx[i]], [y[i], y[i]+Vy[i]])
    return [P]
    

kino = FuncAnimation(fig, TMoM, frames = len(t), interval = 20)

plt.show()