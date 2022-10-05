#This is a utility file for handling data for the NN
#it has several useful functions, as well as two for loading a specific comma-separated formatting
#Use these as a basis for your own!

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


#define solar metallicities
#Ax = log(nx/nh)+12
#so 10^ (Ax-12) = nx/nh
solar = {}
solar['H'] = 12.0
solar['He/H'] = np.power(10,10.93-12)
solar['C/H'] = np.power(10,8.43-12)#12*
solar['N/H'] = np.power(10,7.83-12)#14*
solar['O/H'] = np.power(10,8.69-12) #16*
solar['Ne/H'] = np.power(10,7.93-12)#20*
solar['Mg/H'] = np.power(10,7.60-12)#24*
solar['Si/H'] = np.power(10,7.51-12)#28*
solar['Fe/H'] = np.power(10,7.50-12) #56*
solar['alpha/Fe'] = (solar['O/H'] + solar['Mg/H'] + solar['Si/H'])/(3.0*solar['Fe/H'])

#a coeff to specific lookback time
def a_to_t(a,h=0.6777,Om=0.307):
    H0 = 100*h
    z = 1./a - 1.
    H0time = H0 * (1.0 / 3.086e19) * 3.16e7 * 1e9
    theta = np.sqrt(1-Om) * np.power(Om*np.power(1.+z,3.)+(1-Om), -0.5)
    t = (1./H0time) * (1./(3.*np.sqrt(1-Om))) * np.log((theta+1)/(1-theta))
    return t

#Calculates the inertia tensor of a cloud of vectors
#Essentially fitting a 3D ellipse around the data, with
#major, minor, and intermediate axes.
#these can be treated as eigenvectors,
#and define a basis:
#
#i_tens,eig_val,eig_vec  = utils.inertia_tensor(_xyz,normed=True)
#_xyz = _xyz.dot(eig_vec)
#
#will transform the x,y,z coordinates to that basis set,
#where the direction out of the disc is "up", or 'z'
#
#One big caveat, however, is that while the transformed positions will
#"look" nice, it's *not* the J of the disc! It's purely based on 
#particle positions, not kinematics.
def inertia_tensor(pos,normed=False):
    ds = np.linalg.norm(pos,axis=1)
    if normed:
        ds2 = np.asarray((ds*ds))
    else:
        ds2 = np.ones(len(ds))
    f = (ds2 > 0)
    ds2 = ds2[f]
    xs = np.asarray(pos[:,0][f])
    ys = np.asarray(pos[:,1][f])
    zs = np.asarray(pos[:,2][f])
    i_tens = np.zeros((3,3))
    i_tens[0,0] = np.sum(xs*xs/(ds2))
    i_tens[1,1] = np.sum(ys*ys/(ds2))
    i_tens[2,2] = np.sum(zs*zs/(ds2))

    xy = np.sum(xs*ys/(ds2))
    i_tens[1,0] = xy
    i_tens[0,1] = xy

    xz = np.sum(xs*zs/(ds2))
    i_tens[0,2] = xz
    i_tens[2,0] = xz

    yz = np.sum(ys*zs/(ds2))
    i_tens[1,2] = yz
    i_tens[2,1] = yz

    eig_val,eig_vec = np.linalg.eig(i_tens)

    return i_tens,eig_val,eig_vec

#Code "stolen" from Kyle Oman years ago, 
#and adapted for my own purposes.
#This calculates the angular momentum vectors,
#and generates the new, rotated positions/velocities,
#as well as the rotation matrix itself.
def L_coordinates(xyz, vxyz, frac=.3, distcut = 20, usedist=False, recenter_v=False, v_frac=.3):
    
    xyz = xyz#-xyz_cent
    
    r = np.sqrt(np.sum(np.power(xyz, 2), axis=1))
    rsort = np.argsort(r,kind='quicksort')
    m = np.ones(len(r))	
    p = vxyz #linear momentum
    L = np.cross(xyz, p) #angular momentum
    p = p[rsort]
    L = L[rsort]
    mcumul = np.cumsum(m[rsort]) / np.sum(m)
    if usedist:
        Nfrac = np.argmin(np.abs(r[rsort] - distcut))
        Nvfrac = np.argmin(np.abs(r[rsort] - distcut))
    else:
        Nfrac = np.argmin(np.abs(mcumul - frac))
        Nvfrac = np.argmin(np.abs(mcumul - v_frac))        
    #L = L[:np.floor(len(r) * frac)]
    Nfrac = np.max([Nfrac, 100]) #use a minimum of 100 particles
    Nvfrac = np.max([Nvfrac, 100])
    Nfrac = np.min([Nfrac, len(r)]) #unless this would be more than all particles
    Nvfrac = np.min([Nvfrac, len(r)])
    p = p[:Nfrac]
    L = L[:Nfrac]
    if recenter_v:
        vcent = np.sum(p, axis=0) / np.sum(m[rsort][:Nvfrac])
        vxyz = vxyz - vcent
    Ltot = np.sqrt(np.sum(np.power(np.sum(L, axis=0), 2))) #scalar
    Lhat = np.sum(L, axis=0) / Ltot #unit vector
    #Lhat defines new z direction
    #pick x direction as direction of first gas particle orthogonal to z **could change this to be along semi-major axis or something?
    zhat = Lhat
    xhat = (xyz[0,:] - np.dot(xyz[0,:], zhat) * zhat) #not normalized
    xhat = xhat / np.sqrt(np.sum(np.power(xhat, 2))) #normalized
    yhat = np.cross(zhat, xhat) #guarantees right-handedness
    new_x = np.dot(xyz, xhat)
    new_y = np.dot(xyz, yhat)
    new_z = np.dot(xyz, zhat)
    new_xyz = np.vstack((new_x, new_y, new_z)).T
    new_vx = np.dot(vxyz, xhat)
    new_vy = np.dot(vxyz, yhat)
    new_vz = np.dot(vxyz, zhat)
    new_vxyz = np.vstack((new_vx, new_vy, new_vz)).T
    
    
    rot_matrix = np.vstack((xhat,yhat,zhat))
    return new_xyz, new_vxyz, Lhat, rot_matrix


#An example import function that transforms raw particle data stored in a comma-separated values format
#and makes it usable as input for GANN.
#Interior comments will describe how this makes the data useful, and what sort of data belongs in the CSV
def import_data(colnames=['X','Y','Z','M','R','H','He','C','N','O','Ne','Mg','Si','Fe','epsJ','tform','z','Vr','BE','Lz','pkmassid','acc_label'], to_drop = ['R','epsJ'], dat = False, distcut = 0, alpha_max = 1000, chem_noise = 0):    
    #This is implementation-specific - here I call read_csv on a specific file I've generated from
    #the raw simulation data. 
    #This file has "colnames" stored in ascii as csv, with each line representing each particle
    #The chemical values are stored as the particle mass fractions (easy, since that's the default in the archive files)
    #WITH NO DIVISORS (yet). Technically, you should be able to feed any format you want into the NN, but YMMV.
    #The important values for the NN are:
    #H, O, Mg, Si, Fe, tform
    #corresponding to mass fractions for Hydroden, Oxygen, Magnesium, Silicon, Iron, as well as formation time (in Gyr)
    if(dat):
        dat = pd.read_csv("GalaxyTestData/Au14.tot_dat",na_values = "nan",names=colnames, sep=" ")
    else:        
        dat = pd.read_csv("GalaxyTestData/Au14.TRAINING",na_values = "nan",names=colnames)
    #The physical values (X,Y,Z,M,R,epsJ,z,Vr, BE, Lz) are primarily     
    dataset = dat.copy()
    dataset.tail()
    dataset = dataset.dropna()

    #remove zero metallicity particles
    bad_index = dataset.query("H <= 0 or He <= 0 or C <= 0 or N <= 0 or O <= 0 or Ne <= 0 or Mg <= 0 or Si <= 0 or Fe <= 0")
    #Drop the bad indices
    dataset = dataset.drop(bad_index.index)
    #Cut by radial distance, as a filter
    dataset = dataset.loc[(dataset['R'] > distcut)]
    #get in X/H form
    #First, we do H, since we need it for normalizing.
    #It's also the easiest, since its mass is 1
    dataset['H'] = dataset['H'] * (1+ np.random.standard_normal()*chem_noise)
    #For the others we:
    #              divide by solar-  divide by mass- and grab the value, with the option for added normally-distributed noise 
    dataset['He'] = (1./solar['He/H']) * (1/4.) * (dataset['He'] * (1+ np.random.normal()*chem_noise))/dataset['H']
    dataset['C'] = (1./solar['C/H']) * (1/12.) * (dataset['C'] * (1+ np.random.normal()*chem_noise))/dataset['H']
    dataset['N'] = (1./solar['N/H']) * (1/14.) * (dataset['N'] * (1+ np.random.normal()*chem_noise))/dataset['H']
    dataset['O'] = (1./solar['O/H']) * (1/16.) * (dataset['O'] * (1+ np.random.normal()*chem_noise))/dataset['H']
    dataset['Ne'] = (1./solar['Ne/H']) * (1/20.) * (dataset['Ne'] * (1+ np.random.normal()*chem_noise))/dataset['H']
    dataset['Mg'] = (1./solar['Mg/H']) * (1/24.) * (dataset['Mg'] * (1+ np.random.normal()*chem_noise))/dataset['H']
    dataset['Si'] = (1./solar['Si/H']) * (1/28.) * (dataset['Si'] * (1+ np.random.normal()*chem_noise))/dataset['H']
    dataset['Fe'] = (1./solar['Fe/H']) * (1/56.) * (dataset['Fe'] * (1+ np.random.normal()*chem_noise))/dataset['H']
    #Here we define alpha/Fe, which we need for Auriga data (also Bovy et al., 2017 form)
    dataset['alpha_Fe'] = ((dataset['O']/dataset['Fe'] + dataset['Mg']/dataset['Fe'] + dataset['Si']/dataset['Fe'])/3.0)

    #This is required, really. alpha/Fe has the potential to get *really* large (10^10) for simulated data,
    #since we can have near-zero metal stars, for which the Fe divisor is too small.
    dataset = dataset.loc[dataset['alpha_Fe'] < alpha_max]
    #normalizing by H will reduce all fractions and distributions to a nice 0-1 form.
    
    
    dataset['tform'] = (dataset['tform'])/13.8 #normalize t_form -- likely unnecessary with batchnorm in NN
    #Finally, remove unimportant dimensions from dataset
    #If you want to isolate these, you can do 
    #V = dataset.pop(COLNAME)
    #where V will be the one-column dataset that you just removed
    for weasel in to_drop:
        dataset.pop(weasel)
    
    return dataset