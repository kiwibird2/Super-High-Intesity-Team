# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 16:21:14 2022

@author: ezra.harris
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import *



from matplotlib import rc
rc('font', family='serif')
rc('lines', linewidth=1.5)
rc('font', size=14)
#plt.rc('legend',**{'fontsize':11})



    
def func_flux(q,gamma):
    # Primitive variables
    r=q[0];
    u=q[1]/r;
    E=q[2]/r;
    p=(gamma-1.)*r*(E-0.5*u**2);
    
    # Flux vector
    F0 = np.array(r*u)
    F1 = np.array(r*u**2+p)
    F2 = np.array(u*(r*E+p))
    flux=np.array([ F0, F1, F2 ])
    
    return (flux)

def flux_roe(q,dx,gamma,a,nx):

    # Compute primitive variables and enthalpy
    r=q[0];
    u=q[1]/r;
    E=q[2]/r;
    p=(gamma-1.)*r*(E-0.5*u**2);
    htot = gamma/(gamma-1)*p/r+0.5*u**2
    
    # Initialize Roe flux
    Phi=np.zeros((3,nx-1))
    
    for j in range (0,nx-1):
    
        # Compute Roe averages
        R=sqrt(r[j+1]/r[j]);                          # R_{j+1/2}
        rmoy=R*r[j];                                  # {hat rho}_{j+1/2}
        umoy=(R*u[j+1]+u[j])/(R+1);                   # {hat U}_{j+1/2}
        hmoy=(R*htot[j+1]+htot[j])/(R+1);             # {hat H}_{j+1/2}
        amoy=sqrt((gamma-1.0)*(hmoy-0.5*umoy*umoy));  # {hat a}_{j+1/2}
        
        # Auxiliary variables used to compute P_{j+1/2}^{-1}
        alph1=(gamma-1)*umoy*umoy/(2*amoy*amoy);
        alph2=(gamma-1)/(amoy*amoy);

        # Compute vector (W_{j+1}-W_j)
        wdif = q[:,j+1]-q[:,j];
        
        # Compute matrix P^{-1}_{j+1/2}
        Pinv = np.array([[0.5*(alph1+umoy/amoy), -0.5*(alph2*umoy+1/amoy),  alph2/2],
                        [1-alph1,                alph2*umoy,                -alph2 ],
                        [0.5*(alph1-umoy/amoy),  -0.5*(alph2*umoy-1/amoy),  alph2/2]]);
                
        # Compute matrix P_{j+1/2}
        P    = np.array([[ 1,              1,              1              ],
                        [umoy-amoy,        umoy,           umoy+amoy      ],
                        [hmoy-amoy*umoy,   0.5*umoy*umoy,  hmoy+amoy*umoy ]]);
        
        # Compute matrix Lambda_{j+1/2}
        lamb = np.array([[ abs(umoy-amoy),  0,              0                 ],
                        [0,                 abs(umoy),      0                 ],
                        [0,                 0,              abs(umoy+amoy)    ]]);
                      
        # Compute Roe matrix |A_{j+1/2}|
        A=np.dot(P,lamb)
        A=np.dot(A,Pinv)
        
        # Compute |A_{j+1/2}| (W_{j+1}-W_j)
        Phi[:,j]=np.dot(A,wdif)
        
    #==============================================================
    # Compute Phi=(F(W_{j+1}+F(W_j))/2-|A_{j+1/2}| (W_{j+1}-W_j)/2
    #==============================================================
    F = func_flux(q,gamma);
    Phi=0.5*(F[:,0:nx-1]+F[:,1:nx])-0.5*Phi
    
    dF = (Phi[:,1:-1]-Phi[:,0:-2])
    
    return (dF)

# Parameters
CFL    = 0.50               # Courant Number
gamma  = 1.4                # Ratio of specific heats
ncells = 400                # Number of cells
x_ini =0.; x_fin = 1.       # Limits of computational domain
dx = (x_fin-x_ini)/ncells   # Step size
nx = ncells+1               # Number of points
x = np.linspace(x_ini+dx/2.,x_fin,nx) # Mesh

# Build IC
r0 = zeros(nx)
u0 = zeros(nx)
p0 = zeros(nx)
halfcells = int(ncells/2)

#inital conditions (SOD shock)
p0[:halfcells] = 1.0  ; p0[halfcells:] = 0.1;
u0[:halfcells] = 0.0  ; u0[halfcells:] = 0.0;
r0[:halfcells] = 1.0  ; r0[halfcells:] = 0.125;
tEnd = 0.20;


E0 = p0/((gamma-1.)*r0)+0.5*u0**2 # Total Energy density
a0 = sqrt(gamma*p0/r0)            # Speed of sound
q  = np.array([r0,r0*u0,r0*E0])   # Vector of conserved variables


# Solver loop
t  = 0
it = 0
a  = a0
dt=CFL*dx/max(abs(u0)+a0)         # Using the system's largest eigenvalue

while t < tEnd:

    q0 = q.copy();
    dF = flux_roe(q0,dx,gamma,a,nx);
    
    q[:,1:-2] = q0[:,1:-2]-dt/dx*dF;
    q[:,0]=q0[:,0]; q[:,-1]=q0[:,-1]; # Dirichlet BCs
    
    # Compute primary variables
    rho=q[0];
    u=q[1]/rho;
    E=q[2]/rho;
    p=(gamma-1.)*rho*(E-0.5*u**2);
    a=sqrt(gamma*p/rho);
    if min(p)<0: print ('negative pressure found!')
    
    # Update/correct time step
    dt=CFL*dx/max(abs(u)+a);
    
    # Update time and iteration counter
    t=t+dt; it=it+1;

nNodes = 401
impedance0 = 377.0
maxTime = 401
ez = np.zeros([nNodes])
hy = np.zeros([nNodes])
time = np.linspace(0, maxTime, maxTime)
pos = np.linspace(0, nNodes, nNodes)


J = rho * u

for i in range(len(J)):
    ez[i] = J[i]
                  
for i in range(maxTime):
    for n in range(nNodes):
        if n == nNodes -1:
            break
        ez[n] = ez[n] + (hy[n] - hy[n-1]) 
        hy[n] = hy[n] + (ez[n+1] - ez[n]) / impedance0
        

plt.plot(time, rho)
plt.plot(time, J)
plt.plot(time, ez)

def source(ez, hy, rho, mom):
    dmomndt = rho * (ez + mom*hy)
    denergydt = mom * ez
    dmomndt = dmomndt.T
    denergydt = denergydt.T
    vector = np.column_stack((dmomndt,denergydt))
    return vector

def RK4(f,r,h):
    k1 = h*f(r)
    k2 = h*f(r+.5*k1)
    k3 = h*f(r+.5*k2)
    k4 = h*f(r+k3)
    return r + (k1+2*k2+2*k3+k4)/6
    
def RK4integrate(f, y0, tspan):
    u = np.zeros([len(tspan),len(y0)])
    u[0,:]=y0
    for k in range(1, len(tspan)):
        u[k,:] = RK4(f, u[k-1], tspan[k]-tspan[k-1])
    return u

initialArray = np.zeros([1,4])
y0= np.array([0,0,0,0])
for i in range(len(y0)):
    initialArray[0,:] = y0[i]



# solArray = np.zeros([ncells+1, 2])
# for i in range(len(solArray[:,1])):
#     solArray[i,:] = RK4integrate(source, y0, time)
    

