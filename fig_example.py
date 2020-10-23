# -*- coding: utf-8 -*-
"""
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

Visualisation: Example figure (Fig.5)

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from results_util import comp_pr,comp_roc

def show_example(name_frame='F(1a)',name_scene='E(1)',N=1000,N_supression=0):
    """
    Plots the GT, output and ,,binary'' colormap for the example figure
    Fig.5
    
    parameters:
        name_frame: name of the source image (inductive scenario)
        name_scene: name of the output image
        N: no. vectors in the 2nd stage as int or percentage string e.g. '33p'
        N_supression: no. supressed vectors (extension of the algorithm, unused)
    """
    for i_f,f in enumerate(['scene','frame','exbl']):
        plt.rcParams.update({'font.size': 14})
        fname = "res/{}_{}_{}_{}_{}.npz".format(name_frame,name_scene,N,N_supression,f)
        print (fname) 
        res = np.load(fname)
        pred = res['pred_2'] if f=='exbl' else res['pred_1']
        n_t = np.count_nonzero(res['anno']==1)
        pp = np.ravel(pred)
        arg = np.argsort(pp)[::-1]
        tresh = pp[arg][n_t]
        truth = np.zeros_like(res['anno'])
        truth[res['anno']==1]=1
        bin_res = np.zeros_like(res['anno'])
        bin_res[pred>tresh]=1
        
        final_res = np.zeros_like(res['anno'])
        final_res[np.logical_and(bin_res==1,truth==1)]=1  #TP
        final_res[np.logical_and(bin_res==0,truth==1)]=2  #FN
        final_res[np.logical_and(bin_res==1,truth==0)]=3  #FP
        
        colors = 'white red #3399ff #d3d3d3'.split()
        cmap = ListedColormap(colors, name='colors', N=4)
        
        plt.imsave('res\example_bin_{}.pdf'.format(f),final_res,cmap=cmap)
        
        if f == 'scene':
            ax = plt.subplot()
            plt.rcParams.update({'font.size': 14})
            im = plt.imshow(res['pred_1'],cmap='Reds',interpolation='nearest',aspect='auto')
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.02)
            plt.colorbar(im,cax=cax)
            plt.tight_layout()
            plt.savefig("res\example_pred.pdf",bbox_inches='tight',pad_inches=0)
            plt.close()
        
        plt.rcParams.update({'font.size': 14})
        anno=res['anno']
        anno[anno==15]=0

        colors= ['white','red']
        for i in [0,1,2,3,4,5,6,7,8,9]:
            colors.append(plt.get_cmap('tab10')(i/10))
        cmap = ListedColormap(colors, name='colors', N=len(colors))
        plt.imsave('res\example_gt.pdf',anno,cmap=cmap)
        
def show_stats(name_frame='F(1a)',name_scene='E(1)',N=1000,N_supression=0,is_pr=True,show=True):
    """
    Plots pr and roc curves for an example figure, Fig.5
    
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
    labels=['Ideal MF','Inductive MF','MF']
    for i_f,f in enumerate(['scene','frame','exbl']):
        fname = "res/{}_{}_{}_{}_{}.npz".format(name_frame,name_scene,N,N_supression,f)
        print (fname) 
        res = np.load(fname)
        pred = res['pred_1'] if f=='exbl' else res['pred_1']
            
        if is_pr:
            vauc,precision,recall = comp_pr(pred,res['anno'])
            plt.plot(recall, precision,label="{}, AUC:{:0.2f}".format(labels[i_f],vauc),alpha=0.7,marker=markers[i_f],markevery=0.3,lw=2)
        else:    
            vauc,fpr,tpr  = comp_roc(pred,res['anno'])
            plt.plot(fpr, tpr,label="{}, AUC:{:0.2f}".format(labels[i_f],vauc),alpha=0.7,marker=markers[i_f],markevery=0.3,lw=2)
    
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
        plt.savefig("res\example_{}.pdf".format('pr' if is_pr else 'roc'),bbox_inches='tight',pad_inches=0)
    plt.close()

if __name__ == "__main__":
    
    show_example()
    show_stats(is_pr=True)
