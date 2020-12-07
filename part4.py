#%%

import neutrino_functions as f
import matplotlib.pyplot as plt
import numpy as np 

#4.1 The Univariate Method

diffSqrMass=np.linspace(1e-3,4.8e-3,200)
mixAng=np.linspace(np.pi/32,np.pi/2,200)

fig = plt.figure("2d")
ax = plt.axes()
X, Y = np.meshgrid(mixAng, diffSqrMass)
NLLvals=f.NLL(X,Y,'threedimension')
plt.title("Contour plot of likelihood")
ax.contour(X, Y, NLLvals, 50, cmap='binary')

print(f.univariate(NLL,[mixAng,diffSqrMass]))



#%%
#4.2 Minimise simultaneously 

#Newtons Method

