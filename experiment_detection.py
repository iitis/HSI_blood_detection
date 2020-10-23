# -*- coding: utf-8 -*-
'''
************************************************************************
Copyright 2020 Institute of Theoretical and Applied Informatics, 
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
author: M. Romasewski, mromaszewski@iitis.pl

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
************************************************************************

Code for experiments in the paper by M. Romaszewski, P.Glomb, M. Cholewa, A. Sochan  
,,A Dataset for Evaluating Blood Detection in Hyperspectral Images''
preprint: arXiv:2008.10254

Code for detection experiments (neccesary to generate results) 
'''

import numpy as np
from two_stage_detector import TwoStageMatchedFilter
from os.path import isfile
from ds_load import load_ds, load_exbl, IMAGES
from _ctypes import ArgumentError



def decode_N(N,n_blood):
    """
    Decodes vector length value, turning it into int
    allows to use ints or ,,percentage strings'' e.g. "33p" for 33/100*n_blood
    parameters:
        N: int or a percentage string e,g, '33p' for 33%
        n_blood: a number of all vectors
    returns:
        A number of selected vectors as int    
    """
    if isinstance(N,int):
        assert N>0
        return N
    if not N.endswith('p'):
        raise ArgumentError
    ratio = int(N[:-1])/100.0
    assert (ratio>0 and ratio<=100)
    return int(n_blood*ratio) 

def experiment(name_frame,name_scene,id_exbl,N=1000,N_supression=0):
    """
    Main detection experiment, applies the TSMF 
    in `transductive', `inductive' and `ideal' settings.
    
    parameters:
        name_frame: name of the source image (inductive scenario)
        name_scene: name of the output image
        id_exbl: id of the reference spectrum
        N: no. vectors in the 2nd stage as int or percentage string e.g. '33p'
        N_supression: no. supressed vectors (extension of the algorithm, unused)
    """    
    fname = "res_scenes/{}_{}_{}_{}_{}.npz".format(name_frame,name_scene,N,N_supression,'scene')
    if isfile(fname):
        print (fname, "exists")
        return False
    
    data_frame,anno_frame = load_ds(name_frame)
    data_scene,anno_scene = load_ds(name_scene)
    
    blood_exbl = load_exbl(id_exbl)

    blood_frame = np.mean(data_frame[anno_frame==1],axis=0)
    blood_scene = np.mean(data_scene[anno_scene==1],axis=0)
    
    assert len(blood_exbl)==len(blood_frame)    
    assert len(blood_exbl)==len(blood_scene)

    names = ['scene','frame','exbl']
    
    n_blood = np.count_nonzero(anno_scene==1)
    N_v = decode_N(N, n_blood)
    for i_blood,blood in enumerate([blood_scene,blood_frame,blood_exbl]):
        X_data = data_scene.reshape(-1,data_scene.shape[2]).copy()
        mf = TwoStageMatchedFilter()
        
        mf.fit(blood, X_data, N=N_v, N_supression=N_supression)
        pred_1 = mf.predict(X_data, stage='first')
        pred_2 = mf.predict(X_data, stage='second')
        fname = "res/{}_{}_{}_{}_{}.npz".format(name_frame,name_scene,N,N_supression,names[i_blood])
        np.savez_compressed(fname,pred_1=pred_1.reshape(anno_scene.shape),pred_2=pred_2.reshape(anno_scene.shape),anno=anno_scene)
    return True


def fire_experiment(N=1000):
    """
    Runs the experiment for all IMAGES
    
    parameters:
        N: no. vectors in the 2nd stage as int or percentage string e.g. '33p'
    """     
    N_supression=0
    for im in IMAGES:
        experiment(name_frame=im['name_frame'],name_scene=im['name_scene'],id_exbl=im['id_exbl'],N=N,N_supression=N_supression)


def fire_sensitivity():
    """
    runs the sensitivity experiment (for Fig.11)
    """
    pairs = [11,4,7,8]
    val_N=[100,250,500,750,1000,1250,1500,1750,2000,3000,4000,5000]
    for p in pairs:
        im = IMAGES[p]
        for N in val_N: 
            experiment(name_frame=im['name_frame'],name_scene=im['name_scene'],id_exbl=im['id_exbl'],N=N,N_supression=0)

if __name__ == "__main__":
    fire_sensitivity()
    if False:
        fire_experiment(N=1000)
