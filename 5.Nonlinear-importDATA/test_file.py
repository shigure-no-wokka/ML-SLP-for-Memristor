import numpy as np
import matplotlib.pyplot as plt
from random import randint
from scipy.optimize import curve_fit

GP = abs(np.loadtxt('zyw-syn3-GP.txt'))
GD = abs(np.loadtxt('zyw-syn3-GD.txt'))
GPmax = max(GP)
GPmin = min(GP)
GDmax = max(GD)
GDmin = min(GD)
GP_new = np.array([(each - GPmin) / (GPmax - GPmin) for each in GP])
GD_new = np.array([(each - GDmin) / (GDmax - GDmin) for each in GD])

ap=1e-2
am=5e-3

bp=2.5
bm=1.25

Gmin=1e-4
Gmax=1

def delta_GP(G):
    return G+ap*np.exp(-bp*(G-Gmin)/(Gmax-Gmin))

def delta_GD(G):
    return G-am*np.exp(-bm*(Gmax-G)/(Gmax-Gmin))

G_P = Gmin
GP_list=[Gmin]
while G_P<Gmax:
    G_P = delta_GP(G_P)
    GP_list.append(G_P)

G_D = Gmax
GD_list=[Gmax]
while G_D>Gmin:
    G_D = delta_GD(G_D)
    GD_list.append(G_D)

plt.plot(GP_list[::int(len(GP_list)/20)], 'o-', label='GP_list 20')
plt.plot(GP_list[::int(len(GP_list)/100)], '-', label='GP_list 100')
plt.plot(GP_new, 'o', label='GP_new')
plt.plot(GD_list[::int(len(GD_list)/20)], 'o-', label='GD_list 20')
plt.plot(GD_list[::int(len(GD_list)/100)], '-', label='GD_list 100')
plt.plot(GD_new, 'o', label='GD_new')
plt.legend()
plt.show()