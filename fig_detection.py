# -*- coding: utf-8 -*-
'''
************************************************************************
Copyright 2020 Institute of Theoretical and Applied Informatics, 
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
author: M. Romaszewszki, mromasewski@iitis.pl

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

Visualisation: detection results in Fig.6-8, 11

'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from results_util import comp_pr,comp_roc
from matplotlib.colors import ListedColormap
from experiment_detection import IMAGES

def show_coloured_result(name_frame,name_scene,code,N=1000,N_supression=0,show=True):
    """
    plots coloured detection map (Fig. 6,7,8)
    parameters:
        name_frame: name of the source image (inductive scenario)
        name_scene: name of the output image
        code: code for results file name
        N: no. vectors in the 2nd stage as int or percentage string e.g. '33p'
        N_supression: no. supressed vectors (extension of the algorithm, unused)
        show: True for plot.plot(), False for imsave
    """
    results = {}
    for i_f,f in enumerate(['scene','frame','exbl']):
        fname = "res/{}_{}_{}_{}_{}.npz".format(name_frame,name_scene,N,N_supression,f)
        res = np.load(fname)
        pred = res['pred_2'] if f in ['exbl'] else res['pred_1']
        
        #create a binary map
        n_t = np.count_nonzero(res['anno']==1)
        pp = np.ravel(pred)
        arg = np.argsort(pp)[::-1]
        tresh = pp[arg][n_t]
        truth = np.zeros_like(res['anno'],dtype=np.int32)
        truth[res['anno']==1]=1
        if i_f == 0:
            results['gt']=truth
        bin_res = np.zeros_like(res['anno'],dtype=np.int32)
        bin_res[pred>tresh]=1        
        results[f]=bin_res

    final_res = np.zeros_like(res['anno'])
    final_res[np.logical_and(results['gt']==1,results['exbl']==1)]=1                                       #TP (red)
    final_res[np.logical_and(np.logical_and(results['gt']==1,results['exbl']==0),results['scene']==1)]=2  #FN1 (orange)
    final_res[np.logical_and(np.logical_and(results['gt']==1,results['exbl']==0),results['scene']==0)]=3  #FN2 (blue, not possible) 
    final_res[np.logical_and(results['gt']==0,results['exbl']==1)]=4                                       #FP (light grey)
    final_res[np.logical_and(np.logical_and(results['gt']==0,results['exbl']==0),results['scene']==1)]=5  #FP by ideal but mot exbl (green)

    colors = 'white red orange blue #d3d3d3 green'.split()
    cmap = ListedColormap(colors, name='colors',N=6)
    if show:
        plt.show()
    else:
        plt.imsave('res/res_{}_{}_bin.pdf'.format(code,N),final_res,cmap=cmap)
    plt.close()
      


def show_sensitivity_analysis(N_supression=0,is_pr=True,show=True):
    """
    Graph of the impact of parameter N (no. vectors) , Fig.11
    parameters:
        N_supression: no. supressed vectors (extension of the algorithm, unused)
        is_pr: use PR (True) or ROC (FALSE) AUC
        show: True for plot.plot(), False for imsave    
    """
    images_n = [IMAGES[i] for i in [11,4,7,8]]
    markers = ['s','v','<','>','1','.','p']
    plt.rcParams.update({'font.size': 14})
    val_N=[100,250,500,750,1000,1250,1500,1750,2000,3000,4000,5000]
    for i_im,im in enumerate(images_n):
        nn=[]
        for N in val_N:
            name_frame = im['name_frame']
            name_scene = im['name_scene']
            fname = "res/{}_{}_{}_{}_{}.npz".format(name_frame,name_scene,N,N_supression,"exbl")
            res = np.load(fname)
            pred = res['pred_2']
            if is_pr:
                vauc,_,_ = comp_pr(pred,res['anno'])
                nn.append(vauc)
            else:    
                vauc,_,_  = comp_roc(pred,res['anno'])
                nn.append(vauc)
        plt.plot(val_N,nn,label=im['code'],marker=markers[i_im],markevery=2)
    if is_pr:
        plt.ylabel('AUC(PR)')
    else:
        plt.ylabel('AUC(ROC)')
    
    plt.xlabel('N')
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('res/san_{}_{}.pdf'.format(N,"pr" if is_pr else "roc"),bbox_inches='tight',pad_inches=0)
    plt.close()

    
def show_stats(name_frame='F(1)',name_scene='E(1)',code='E(1)',N=1000,N_supression=0,is_pr=True,show=True):
    """
    plots pr and roc curves for a given detection Fig.6-8
    parameters:
        name_frame: name of the source image (inductive scenario)
        name_scene: name of the output image
        code: code for results file name
        N: no. vectors in the 2nd stage as int or percentage string e.g. '33p'
        N_supression: no. supressed vectors (extension of the algorithm, unused)
        is_pr: use PR (True) or ROC (FALSE) AUC
        show: True for plot.plot(), False for imsave    
    
    """
    plt.rcParams.update({'font.size': 14})
    markers = ['s','v','<','>','1','.','p']
    labels=['Ideal MF','Inductive MF','Algorithm 1']
    for i_f,f in enumerate(['scene','frame','exbl']):
        fname = "res/{}_{}_{}_{}_{}.npz".format(name_frame,name_scene,N,N_supression,f)
        print (fname) 
        res = np.load(fname)
        pred = res['pred_2'] if f in ['exbl'] else res['pred_1']
        
        if is_pr:
            vauc,precision,recall = comp_pr(pred,res['anno'])
            plt.plot(recall, precision,label="{} AUC:{:0.2f}".format(labels[i_f],vauc),alpha=0.7,marker=markers[i_f],markevery=0.3,lw=2)
        else:    
            vauc,fpr,tpr  = comp_roc(pred,res['anno'])
            plt.plot(fpr, tpr,label="{} AUC:{:0.2f}".format(labels[i_f],vauc),alpha=0.7,marker=markers[i_f],markevery=0.3,lw=2)
   
    
    if is_pr:
        plt.ylabel('Precision')
        plt.xlabel('Recall')
    else:
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--',alpha=0.1)
                
    plt.legend()
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('res/res_{}_{}_{}.pdf'.format(code,N,'pr' if is_pr else 'roc'),bbox_inches='tight',pad_inches=0)
    plt.close()


def show_table_row(name_frame,name_scene,code,N=1000,N_supression=0):
    """
    plots rows of the result table (in LaTeX), Tab.3
    parameters:
        name_frame: name of the source image (inductive scenario)
        name_scene: name of the output image
        code: code for results file name
        N: no. vectors in the 2nd stage as int or percentage string e.g. '33p'
        N_supression: no. supressed vectors (extension of the algorithm, unused)
    """
    str = ""
    str+="{}&".format(code)
    for i_f,f in enumerate(['scene','frame','exbl']):
        fname = "res/{}_{}_{}_{}_{}.npz".format(name_frame,name_scene,N,N_supression,f)
        res = np.load(fname)
        vauc,_,_ = comp_pr(res['pred_1'],res['anno'])
        str+="{:0.2f}&".format(vauc)
        if f == "exbl":
            vauc,_,_ = comp_pr(res['pred_2'],res['anno'])
            str+="{:0.2f}\\NN".format(vauc)
    print (str)

def compare_in_columns(Ns=[750,1000,'10p','20p']):
    """
    plots rows of the result table (in LaTeX), Tab.4
    compare results for different values of N
    parameters:
    Ns: list values of N to plot for    
    """
    for im in IMAGES:
        ss="{}".format(im['code'].replace("_","\_"))
        for N in Ns:
            fname = "res/{}_{}_{}_{}_{}.npz".format(im['name_frame'],im['name_scene'],N,0,'exbl')
            res = np.load(fname)
            vauc,_,_ = comp_pr(res['pred_2'],res['anno'])
            ss+="&{:0.2f}".format(vauc)
        ss+='\\NN'
        print (ss)    

def show_table(N=1000,N_supression=0):
    """
    plots the entire results table (in LaTeX), Tab.3
    parameters:
        N: no. vectors in the 2nd stage as int or percentage string e.g. '33p'
        N_supression: no. supressed vectors (extension of the algorithm, unused)    
    """
    for im in IMAGES:
        show_table_row(name_frame=im['name_frame'],name_scene=im['name_scene'],code=im['code'],N=N,N_supression=N_supression)

if __name__ == "__main__":
    N = 1000
    N_supression=0
    show_sensitivity_analysis()
    show_table(N=1000,N_supression=0)
    for im in IMAGES:
        if True:
            show_coloured_result(name_frame=im['name_frame'],name_scene=im['name_scene'],code=im['code'],N=N,N_supression=N_supression)
        if True:
            show_stats(name_frame=im['name_frame'],name_scene=im['name_scene'],code=im['code'],N=N,N_supression=N_supression)
        sys.exit()
