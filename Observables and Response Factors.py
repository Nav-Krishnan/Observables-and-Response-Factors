# -*- coding: utf-8 -*-
"""
@author: Navneet Krishnan, Australian National University, Department of Theortical Physics
"""


'''Import required libraries'''

from scipy.io import FortranFile
import numpy as np
import math as m
from sympy import symbols, integrate

'''Import WFfile.'''
name = input("Please enter a file name: ")
dat = FortranFile(name,'r') 

'''Read initial records'''

rec1 = dat.read_ints(dtype=np.int32) #Input data
rec2 = dat.read_ints(dtype=np.int32) #Mesh Size
rec3 = dat.read_reals(dtype=float) #Coordinates
rec4 = dat.read_reals(dtype=float) #More data, including state occupation numbers
rec5 = dat.read_reals(dtype=float) #Bookkeeping


'''Define some needed numbers'''

hbar = 197.3 #Planck's constant in MeV fm/c
mneut = 939.565 #neutron mass in MeV/c^2
mprot = 938.272 #proton mass in MeV/c^2


'''Define coordinate arrays. Parameters nx,dx,ny,dy,nz,dz might need to be changed to fit input to Sky3D code.'''

coords = list(rec3) #Full list of x,y and z coordinates for the mesh

nx = 24  #Number of x values

ny = 24  #Number of y values

nz = 24  #Number of z values

xarr = coords[0:nx] #Array of x coordinates

yarr = coords[nx:nx+ny] #Array of y coordinates

zarr = coords[nx+ny:nx+ny+nz] #Array of z coordinates

dx = 1 #Mesh spacing in x direction

dy = 1 #Mesh spacing in y direction

dz = 1 #Mesh spacing in z direction

w = dx*dy*dz #Volume Element for the mesh


'''Define the particle numbers.'''

nneut = rec1[8] #Neutron number

nprot = rec1[9] #Proton number

nst = nneut + nprot #Particle Number

neutocc = rec4[0:nneut] #Neutron Occupation Numbers

protocc = rec4[nneut:nst] #Proton Occupation Numbers



del coords #Remove a now unecessary variable



'''Read records containing the wavefunctions'''

#Read neutron wavefunctions
WFdict = {}
for i in range(1,nneut+1):
    WFdict[i] = dat.read_reals(dtype=float)
    

    
#Read proton wavefunctions       
for i in range(1,nprot+1):
    WFdict[nneut+i] = dat.read_reals(dtype=float)




'''The wavefunctions are originally stored as arrays of reals. As they're complex-valued,
the real and imaginary parts are stored as two separate numbers. 
The next set of code turns a wavefunction into an array of complex numbers for computational ease.'''

#This function takes in a wavefunction of the original form and returns an array of complex values.
def complexify(l): 
    cl = []
    i=0
    while i <= len(l)-2:
        cl.append(complex(l[i],l[i+1]))
        i = i+2
    return(cl)


    
CWFdict = {} #Defines a dictionary to store the complexified wavefunctions


#The loop complexifies all of the wavefunctions and stores them in the dictionary
for i in range(1,nst+1):
    CWFdict[i] = complexify(WFdict[i])

    
del WFdict #Deletes the now unneeded original arrays




'''The wavefunctions are orignally stored in a somewhat confusing order. 
The next function rearranges a wavefunction so they're indexed in the order (spin, z coordinate, y coordinate, x coordinate). 
Spin index is 0 for spin-up, 1 for spin-down.'''

#Define rearrangeing function
def rearrange(WF):
    rearr = []
    for l in range(0,2):
        rearr.append([])
        for k in range(0, len(zarr)):
            rearr[l].append([])
            for j in range(0, len(yarr)):
                rearr[l][k].append([])
                for i in range(0,len(xarr)):
                    rearr[l][k][j].append(WF[13824*l + k*24**2 + j*24 + i])
    return(np.array(rearr))

Rdict = {} #Define a dictionary for the rearranged WFs

#Construct dictionary of rearranged wavefunctions
for i in range(1,nst+1):
    Rdict[i] = np.array(rearrange(CWFdict[i]))
    

    
CWFdict #Deletes the older dictionary




    

    

''' This function determines the norm of a wavefunction.'''

def norm2(WF):
    norm = sum(sum(sum(sum(WF*WF.conjugate()*w))))
    return norm.real

'''The next three functions define an inner product. Vector operators such as spin have
components for each coordinate, and their output is stored as a vector of wavefunctions. 
The vecinnerprod function takes in a single wavefunction for the first argument, and a vector for the second.
It returns a vector of inner products (this is useful for calculating the expectation values of vector operators). 
The innerproduct function determines whether the input is a single wavefunction or a vector of functions, 
then calls the appropriate inner product function.'''

def scainnerprod(WF1,WF2):
    ip = sum(sum(sum(sum(WF1.conjugate()*WF2*w))))
    return(ip)
    
def vecinnerprod(WF1,WF2):
    ipx = sum(sum(sum(sum(WF1.conjugate()*WF2[0]*w))))
    ipy = sum(sum(sum(sum(WF1.conjugate()*WF2[1]*w))))
    ipz = sum(sum(sum(sum(WF1.conjugate()*WF2[2]*w))))
    
    ip = np.array([ipx,ipy,ipz])
    return(ip)
    
def innerproduct(WF1,WF2):
    if len(WF2) == 2:
        ip = scainnerprod(WF1,WF2)
        return(ip)
    elif len(WF2) == 3:
        ip = vecinnerprod(WF1,WF2)
        return(ip)
    else:
        print('Dimension Error: WF2 does not have recognised wavefunction dimensions.')
        
#The inner product function returns a Dimension Error if the input WF2 is neither a single WF or a vector of three WFs.
    
    

'''The next set of code sets up everything needed to take partial derivatives of the wavefunctions.'''

'''Define the Fourier space grid'''

dkx = 2*np.pi/(nx*dx) #kx spacing

dky = 2*np.pi/(ny*dy) #ky spacing

dkz = 2*np.pi/(nz*dz) #kz spacing


#Construct the kx array
kxarr = []
for n in range(1,int(nx/2 + 1)):
    kxarr.append((n-1)*dkx)
for n in range(int(nx/2+1),nx+1):
    kxarr.append((n-nx-1)*dkx)
kxarr = np.array(kxarr)


#Construct the ky array
kyarr = []   
for n in range(1,int(ny/2 + 1)):
    kyarr.append((n-1)*dky)
for n in range(int(ny/2+1),ny+1):
    kyarr.append((n-ny-1)*dky)
kyarr = np.array(kyarr)


#Construct the kz array   
kzarr = []   
for n in range(1,int(nz/2 + 1)):
    kzarr.append((n-1)*dkz)
for n in range(int(nz/2+1),nz+1):
    kzarr.append((n-nz-1)*dkz)
kzarr = np.array(kzarr)


    

'''The next segment defines the matrices used to calculate the derivatives.'''


#Construct the first partial derivative matrix in the x direction.
Dxmat = []

for v in range(0,nx):
    Dxmat.append([])
    for u in range(0,nx):
        tot = 0
        for n in range(0,len(kxarr)):
            loc = np.exp(1j*kxarr[n]*xarr[v])*1j*kxarr[n]*np.exp(-1j*kxarr[n]*xarr[u])
            tot += loc
        Dxmat[v].append( (1/nx)*tot)
Dxmat=np.array(Dxmat)   



#First partial derivative matrix in the y direction.    
Dymat = []

for v in range(0,ny):
    Dymat.append([])
    for u in range(0,ny):
        Dymat[v].append( (1/ny)*sum(np.exp(1j*kyarr*yarr[v])*1j*kyarr*np.exp(-1j*kyarr*yarr[u])))
        
    
#First partial derivative in the z direction.     
Dzmat = []

for v in range(0,nz):
    Dzmat.append([])
    for u in range(0,nz):
        Dzmat[v].append( (1/nz)*sum(np.exp(1j*kzarr*zarr[v])*1j*kzarr*np.exp(-1j*kzarr*zarr[u])))
Dzmat = np.array(Dzmat)




'''The next three loops construct the second-order partial derivative matrices'''


#Second x partial derivative matrix.  
Dx2mat = []

for v in range(0,nx):
    Dx2mat.append([])
    for u in range(0,nx):
        tot = 0
        for n in range(0,len(kxarr)):
            loc = np.exp(1j*kxarr[n]*xarr[v])*((1j*kxarr[n])**2)*np.exp(-1j*kxarr[n]*xarr[u])
            tot += loc
        Dx2mat[v].append( (1/nx)*tot)
Dx2mat=np.array(Dx2mat)  


#Second y partial derivative matrix
Dy2mat = []

for v in range(0,ny):
    Dy2mat.append([])
    for u in range(0,ny):
        Dy2mat[v].append( (1/ny)*sum(np.exp(1j*kyarr*yarr[v])*((1j*kyarr)**2)*np.exp(-1j*kyarr*yarr[u])))
Dy2mat=np.array(Dy2mat)


#Second z partial derivative matrix      
Dz2mat = []

for v in range(0,nz):
    Dz2mat.append([])
    for u in range(0,nz):
        Dz2mat[v].append( (1/nz)*sum(np.exp(1j*kzarr*zarr[v])*((1j*kzarr)**2)*np.exp(-1j*kzarr*zarr[u])))
Dz2mat=np.array(Dz2mat)



'''These define a list of derivative matrices for each coordinate direction.'''

matxlist = [Dxmat, Dx2mat]

matylist = [Dymat, Dy2mat]

matzlist = [Dzmat, Dz2mat]



'''The next set defines the partial derivative functions. They take as input a wavefunction and an order of partial derivative,
and output the differentiated wavefunction. As written, this only works for first and second derivatives,
but the extra derivative matrices can be added in above and put into the matlists, allowing higher-order derivatives.'''

def partialderx(WF,order):
    der = []
    
    mat = matxlist[order-1]

    for l in range(0,2):  
        der.append([])
        for k in range(0,len(zarr)):
            der[l].append([])
            for j in range(0,len(yarr)):
                der[l][k].append([])
                for i in range(0,len(xarr)):
                    floc = 0
                    for n in range(0,len(xarr)):
                        floc += mat[i][n]*WF[l][k][j][n]
                    der[l][k][j].append(floc)
    return(np.array(der))
    
'''As the wavefunction arrays are indexed with x being the last cooridnate, an appending system
could be used to generate the derivatives. As y and z values need to be indexed differently, 
the derivative functions for these generate empty arrays and then rewrite the values.'''    
    
def partialdery(WF,order):
    der =np.empty([2,24,24,24],dtype=complex)
    
    mat = matylist[order-1]
    
    for l in range(0,2):
        for k in range(0,len(zarr)):
            for i in range(0,len(xarr)):
                for j in range(0,len(yarr)):
                    floc = 0
                    for n in range(0,len(yarr)):
                        floc += mat[j][n]*WF[l][k][n][i]
                    der[l][k][j][i] = floc
    return(der)

def partialderz(WF,order):
    der =np.empty([2,24,24,24],dtype=complex)
    
    mat = matzlist[order-1]
    
    for l in range(0,2):
        for j in range(0,len(yarr)):
            for i in range(0,len(xarr)):
                for k in range(0,len(zarr)):
                    floc = 0
                    for n in range(0,len(zarr)):
                        floc += mat[k][n]*WF[l][n][j][i]
                    der[l][k][j][i] = floc
    return(der)
    
    
'''The next set defines the action of the observable operators and gets their expectation values.
Not all observables are split into one function for the action and one for the expectation: that was only needed for some.
Unless otherwise specified, all take as input a wavefunction.'''


'''Position expectation value'''
def position(WF):
    pos = np.array([0,0,0])
    for l in range(0,2):
        for k in range(0,len(zarr)):
            for j in range(0,len(yarr)):
                for i in range(0,len(xarr)):
                    ploc = np.array([xarr[i],yarr[j],zarr[k]])*(WF[l][k][j][i]*(WF[l][k][j][i].conjugate()))*w
                    pos = pos + ploc.real
    return(pos)


'''Spin'''

#Operator
def spin(WF):
    #Get separate spin WFs
    su = np.array(WF[0])
    sd = np.array(WF[1])
    
    SxWF = hbar*0.5*np.array([sd,su])
    SyWF = hbar*0.5*np.array([-1j*sd,1j*su])
    SzWF = hbar*0.5*np.array([su,-1*sd])
    
    SWF = np.array([SxWF,SyWF,SzWF])
    return(SWF)
    
#Expectation 
def spinexp(WF):
    sp = spin(WF)
    
    Sx = innerproduct(WF,sp[0])
    Sy = innerproduct(WF,sp[1])
    Sz = innerproduct(WF,sp[2])
    return(np.array([Sx,Sy,Sz]).real)
    


'''Pauli matrices'''

#Operator

def sigma(WF):
    #Get seperate pauli WFS
    su = np.array(WF[0])
    sd = np.array(WF[1])
    
    sigxWF = np.array([sd,su])
    sigyWF = np.array([-1j*sd,1j*su])
    sigzWF = np.array([su,-1*sd])
    
    sigWF = np.array([sigxWF, sigyWF,sigzWF])
    return(sigWF)
    
#Expectation
def sigexp(WF):
    sig = sigma(WF)
    
    sigx = innerproduct(WF,sig[0])
    sigy = innerproduct(WF,sig[1])
    sigz = innerproduct(WF,sig[2])
    
    return(np.array([sigx,sigy,sigz]).real)

    

'''These functions were unused in my work, but may be useful: they determine the action of the Pauli matrices in a 
spherical basis.'''

#Operator
def sigma1M(WF):
    sig = sigma(WF)
    
    sig1p = -1*(1/m.sqrt(2))*(sig[0]+1j*(sig[1]))
    sig1m = (1/m.sqrt(2))*(sig[0] - 1j*sig[1])
    sig10 = sig[2]
    
    return(np.array([sig1p, sig10,sig1m]))

#Expectation
def sig1Mexp(WF):
    sig1M = sigma1M(WF)
    
    sig1Mexp = innerproduct(WF,sig1M)
    
    return(sig1Mexp)
    
    
'''Momentum'''

#Operator
def momentum(WF):
    mom = []
    mom.append(-1j*hbar*partialderx(WF,1))
    mom.append(-1j*hbar*partialdery(WF,1))
    mom.append(-1j*hbar*partialderz(WF,1))
    mom = np.array(mom)
    return(mom)
    
#Expectation
def momexp(WF):
    mom = momentum(WF)
    expx = innerproduct(WF,mom[0])
    expy = innerproduct(WF,mom[1])
    expz = innerproduct(WF,mom[2])
    exp = np.array([expx,expy,expz])
    return(exp.real)
    
    
'''Kinetic energy. Takes as input the wavefucntion and particle mass.'''

#Operator    
def kinetic(WF,m):
    kin = []
    kin.append(partialderx(WF,2))
    kin.append(partialdery(WF,2))
    kin.append(partialderz(WF,2))
    kin = np.array(kin)
    kin = kin*(-0.5*hbar**2/(m))
    return(kin)
    
#Expectation  
def kinexp(WF,m):
    kin = kinetic(WF,m)
    Ex = innerproduct(WF,kin[0])
    Ey = innerproduct(WF,kin[1])
    Ez = innerproduct(WF,kin[2])
    E = Ex + Ey + Ez
    return(E.real)
    
    
'''Angular momentum'''

#Operator 
def angmomentum(WF):
    mom = momentum(WF)/hbar
    angmom = [[],[],[]]
    for l in range(0,2):
        for n in range(0,len(angmom)):
            angmom[n].append([])
        for k in range(0,len(zarr)):
            for n in range(0,len(angmom)):
                angmom[n][l].append([])
            for j in range(0,len(yarr)):
                for n in range(0,len(angmom)):
                    angmom[n][l][k].append([])
                for i in range(0,len(xarr)):
                    momloc = np.array([mom[0][l][k][j][i], mom[1][l][k][j][i],mom[2][l][k][j][i]])
                    pos = np.array([xarr[i],yarr[j],zarr[k]])
                    angmomloc = np.cross(pos,momloc)
                    for n in range(0,len(angmom)):
                        angmom[n][l][k][j].append(angmomloc[n])
    return(np.array(angmom))

#Expectation
def angmomexp(WF):
    angmom = angmomentum(WF)
    L = innerproduct(WF,angmom)
    return(L.real)
                    
'''Spin orbit expectation value'''

def sl(WF):
    angmom = angmomentum(WF)

    Sx = sigma(angmom[0])[0]
    Sy = sigma(angmom[1])[1]
    Sz = sigma(angmom[2])[2]
    
    tot = Sx + Sy + Sz
    exp = innerproduct(WF,tot)
    return(exp.real)

'''Unused: Total angular mometnum'''    
def totangexp(WF):
    totang = angmomentum(WF) + spin(WF)
    exp = innerproduct(WF,totang)
    
    return(exp.real)



'''This writes a file containing a list of observables for each wavefunction. It takes as input a name for the file, 
and an integer l, and calculates the observables for the first l particles. 
Setting l = nst will generate a list for all the particles.'''

def expwriter(name,l):
    file = open(name + '.txt','w')
    for i in range(1,l+1):
        file.write('WF' + str(i) + '\n')
        file.write('Norm:')
        file.write(str(round(norm2(Rdict[i]),3)))
        file.write('\n')
        file.write('Position Expectation Value:')
        file.write(str(position(Rdict[i])))
        file.write('\n')
        file.write('Pauli Expectation Value:')
        file.write(str(np.round(sigexp(Rdict[i]),3)))
        file.write('\n')
        file.write('Momentum Expectation Value:')
        file.write(str(np.round(momexp(Rdict[i]),3)))
        file.write('\n')
        file.write('Kinetic Energy Expectation Value:')
        if i <= nneut:
            file.write(str(round(kinexp(Rdict[i],mneut),3)))
        else:
            file.write(str(round(kinexp(Rdict[i],mprot),3)))
        file.write('\n')
        file.write('Angular Momentum Expectation Value:')
        file.write(str(np.round(angmomexp(Rdict[i]),3)))
        file.write('\n \n')
    file.close()

'''The next function evaluates the neutron-only and proton-only single response factors. 
It takes as input a name for the output file.'''

def ResponseFactors(name):
    q = symbols('q')
    file = open(name + '.txt','w')
    file.write('Nuclear Response Factors \n \n')
    
    
    O1n = 0
    O2n = 0
    for i in range(1,nneut+1):
        j = i-1
        O1n += norm2(Rdict[i])*neutocc[j]
        O2n += sl(Rdict[i])*neutocc[j]
        
        
    O1p = 0
    O2p = 0
    for i in range(nneut+1,nst+1):
        j= i - nneut - 1
        O1p += norm2(Rdict[i])*protocc[j]
        O2p += sl(Rdict[i])*protocc[j]
    
    
    file.write('1 Neutron: ' + str(O1n) )
    file.write('\n')
    file.write('F1(n,n): '+str(O1n**2))
    file.write('\n')
    file.write('Integrated Response: ' + str(integrate((q/2)*(O1n**2),(q,0,100))))
    file.write('\n')
    file.write('1 Proton: ' + str(O1p))
    file.write('\n')
    file.write('F1(p,p): '+str(O1p**2))
    file.write('\n')
    file.write('Integrated Response: ' + str(integrate((q/2)*(O1p**2),(q,0,100))))
    file.write('\n')
    file.write('F1(p,n): ' + str(O1p*O1n))
    file.write('\n')
    file.write('Integrated Response: ' + str(integrate((q/2)*(O1p*O1n),(q,0,100))))
    file.write('\n \n')
    
    file.write('Spin-orbit Neutron: ' + str(O2n) )
    file.write('\n')
    file.write('F2(n,n): '+str((O2n**2)/9))
    file.write('\n')
    file.write('Integrated Response: ' + str(integrate((q/2)*((O2n**2)/9),(q,0,100))))
    file.write('\n')
    file.write('Spin-orbit Proton: ' + str(O2p))
    file.write('\n')
    file.write('F2(p,p): '+str((O2p**2)/9))
    file.write('\n')
    file.write('Integrated Response: ' + str(integrate((q/2)*((O2p**2)/9),(q,0,100))))
    file.write('\n')
    file.write('F2(p,n): ' + str((O2p*O2n)/9))
    file.write('\n')
    file.write('Integrated Response: ' + str(integrate((q/2)*((O2p*O2n)/9),(q,0,100))) )
    file.write('\n \n')    
    
    file.write('F(1p,2p): ' + str(O1p*O2p/3))
    file.write('\n')
    file.write('Integrated Response: ' + str(integrate((q/2)*(O1p*O2p/3),(q,0,100))))
    file.write('\n')
    file.write('F(1n,2n): ' + str(O1n*O2n/3))
    file.write('\n')
    file.write('Integrated Response: ' + str(integrate((q/2)*(O1n*O2n/3),(q,0,100))))
    file.write('\n')
    file.write('F(1p,2n): ' + str(O1p*O2n/3))
    file.write('\n')
    file.write('Integrated Response: ' + str(integrate((q/2)*(O1p*O2n/3),(q,0,100))))
    file.write('\n')
    file.write('F(1n,2p): ' + str(O1n*O2p/3))
    file.write('\n')
    file.write('Integrated Response: ' + str(integrate((q/2)*(O1n*O2p/3),(q,0,100))))

    
    file.close()
    

  

    
    
