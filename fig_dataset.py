# -*- coding: utf-8 -*-
'''
************************************************************************
Copyright 2020 Institute of Theoretical and Applied Informatics, 
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
author: M. Romaszewski, mromaszewski@iitis.pl

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

Visualisation of dataset properties

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA

from ds_load import load_ds, get_wavelengths
from ds_load import get_rgb as hsi2rgb
from experiment_detection import IMAGES


        
def show_classes(name='F(1)',absorbance=False):
    """
    plots spectra of classes in the image (Fig.1)
    
    Parameters:
    ---------------------
    name: image name
    absorbance: transform reflectance into (pseudo)absorbance i.e. log(1/R)
    """
    data,anno = load_ds(name,normalise=False)
    wav = get_wavelengths()
    plt.rcParams.update({'font.size': 14})

    plt.rcParams.update({'font.size': 12})
    labels = ['blood','ketchup','artificial blood','beetroot juice','poster paint','tomato concentrate','acrylic paint']
    markers = ['s','v','<','>','1','.','p']
    cmap = plt.get_cmap("Set1")
    colors = [cmap(i/7) for i in range(8)]
    
    colors[-4]=cmap(1.0)
    for c_label in range(1,8):
        if c_label in anno: 
            al=1.0 if c_label==1 else 0.7 
            pattern = data[anno==c_label]
            s = np.median(pattern,axis=0)
            if absorbance:
                s[s==0]+=0.0001
                s=np.log10(1.0/s)    
            plt.plot(wav,s,label="{} ({})".format(labels[c_label-1],c_label),color=colors[c_label-1],alpha=al,marker=markers[c_label-1],markevery=10)
    plt.legend()

    if absorbance:
        plt.ylabel("Log(1/R)")
    else:
        plt.ylabel("Reflectance")    
    plt.xlabel("Wavelenghts")
    
    plt.tight_layout(pad=0)
    plt.show()
               
    plt.close()        

def show_mixtures(absorbance=True):
    """
    plots differences of blood spectra on different backgrounds (Fig.2)
    
    Parameters:
    ---------------------
    absorbance: transform reflectance into (pseudo)absorbance i.e. log(1/R)
    """
    wav = get_wavelengths()
    data,gt = load_ds('E(1)',normalise=False)
    
    #lower range and material
    backgrounds = [[44,'metal'],[74,'plastic'],[147,'wood'],[192,'blue'],[271,'red(t-shirt)'],[332,'mixed'],[430,'mixed(green)'],[516,'red(sweater)']]
    
    plt.rcParams.update({'font.size': 14})
    where = gt!=1
    gt[where]=0
    markers = ['s','v','<','>','1','.','p','x']
    for i_b,b in enumerate(backgrounds):
        gta = gt.copy()
        gta[b[0]:,:]=0
        if i_b!=0:
            gta[:backgrounds[i_b-1][0]:,:]=0
        X = data[gta==1,:]
        print (b,X.shape)
        s = np.median(X,axis=0)
        if absorbance:
            s[s==0]+=0.0001
            s=np.log10(1.0/s)    
        plt.plot(wav,s,label="{}".format(b[1]),marker=markers[i_b],markevery=10)
    plt.legend()    
    
    if absorbance:
        plt.ylabel("Log(1/R)")
    else:
        plt.ylabel("Reflectance")    
    plt.xlabel("Wavelenghts")
    plt.show()
    plt.close()

                

def show_days(absorbance=True):
    """
    plots blood spectra sorted by days (Fig.2)

    Parameters:
    ---------------------
    absorbance: transform reflectance into (pseudo)absorbance i.e. log(1/R)
    """
    wav = get_wavelengths()
    
    days = ["Day 1(~1h)","Day 1(~5h)","Day 2","Day 7"]
    frames = ["F(1)","F(1s)","F(2)","F(7)"]
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots()
    markers = ['s','v','<','>','1','.','p']
    
    for i_d,d in enumerate(days):
        data,anno = load_ds(frames[i_d],normalise=False)
        s=np.mean(data[anno==1],axis=0)
        if absorbance:
            s[s==0]+=0.0001
            s=np.log10(1.0/s)    
        plt.plot(wav,s,label="{}".format(d),marker=markers[i_d],markevery=10)
    plt.legend()
    y0,y1 = ax.get_ylim()
    plt.xlim(400,1000)
    if absorbance:
        plt.ylim(0.2,y1)
    
    
    plt.plot([542,542],[0,y1],lw=0.5, linestyle='--',alpha=0.7,color='black')
    plt.annotate('542',xy=(542,y1),xytext=(525, y1),fontsize=10)
    plt.plot([577,577],[0,y1],lw=0.5, linestyle='--',alpha=0.7,color='black')
    plt.annotate('577',xy=(577,y1),xytext=(560, y1),fontsize=10)
    
    if absorbance:
        plt.ylabel("Log(1/R)")
    else:
        plt.ylabel("Reflectance")    
    plt.xlabel("Wavelenghts")
    
    plt.tight_layout()
    plt.show()
    plt.close()


def show_pca(name='F(1)'):
    """
    plots PCA visualisation of data (projection on first PC)
    Fig.2
    
    Parameters:
    ---------------------
    name: image name
    """    
    plt.rcParams.update({'font.size': 14})
    data,gt = load_ds(name,normalise=False)

    data = np.delete(data,445,0)
    gt=np.delete(gt,445,0)
    
    X,y,_ = data2xy(data,gt)

    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)
    
    plt.rcParams.update({'font.size': 12})
    where = y==0
    plt.scatter(X_pca[where,0], X_pca[where,1],c='grey',label='background',marker='1',alpha=0.5)
    where = np.logical_and(np.logical_and(y!=1,y!=8),y!=0)
    plt.scatter(X_pca[where,0], X_pca[where,1],c='blue',alpha=0.5,label='blood-like substances',marker='.')
    where = y==8
    plt.scatter(X_pca[where,0], X_pca[where,1],c='orange',label='uncertain blood',marker='o',alpha=0.5)
    where = y==1
    plt.scatter(X_pca[where,0], X_pca[where,1],c='red',label='blood',marker='x')
    plt.legend()
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.tight_layout(pad=0)
    plt.show()
    plt.close()


def create_gt_images():
    for im in IMAGES:
        print (im['code'])
        data,anno = load_ds(im['name_scene'],normalise=False)
        save_gt(data,anno,im['code'])


def save_gt(data,anno_in,code):
    """
    saves rgb with class annotation
    fig.6-8

    Parameters:
    ---------------------
    data: data cube as nparray
    annotation: 2d annotation array
    """

    ax=plt.subplot()
    colors=[]
    
    #listerd colormap
    for i in [0,6,1,2,3,4,5,10,7,8]:
        colors.append(plt.get_cmap('tab20')(i/20))
    cmap = ListedColormap(colors, name='colors',N=len(colors))
    
    anno=anno_in.copy()
    anno[anno==15]=0
    
    rgb = hsi2rgb(data,wavelengths=get_wavelengths(),gamma=0.7)
    plt.imshow(rgb,aspect='auto')
    X,y,pos = data2xy(data,anno)
    for u in np.unique(y)[::-1]:
        if u!=0:
            where = y==u
            pw = pos[where]
            r = [t[0] for t in pw]
            c = [t[1] for t in pw]
            sc = plt.scatter(x=c, y=r, marker=',', s=3, lw=1,alpha=0.8,color=cmap(u))
    ax.set_axis_off()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout(pad=0)
    plt.savefig('res/res_{}.png'.format(code),bbox_inches='tight',pad_inches=0)
    plt.close()

        
def data2xy(data,truth):
    """
    converts HS cube with truth to X,y,pos

    Parameters:
    ---------------------
    data: data cube as nparray
    annotation: 2d annotation array
    
    Returns:
    -----------------------
    X: 2d array (no. pixels x no.bands)
    y: labels for pixels
    pos: rc position of pixels     
    """
    sh=data.shape
    X=[]
    y=[]
    pos=[]
    for r in range(sh[0]):
        for c in range(sh[1]):
            X.append(data[r,c,:])
            y.append(truth[r,c])
            pos.append([r,c])
    X=np.asarray(X)        
    y=np.asarray(y)
    pos = np.asarray(pos)
    return X,y,pos        

if __name__ == "__main__":
    
    show_classes()
    show_mixtures()
    show_days()
    show_pca()
    create_gt_images()
 
