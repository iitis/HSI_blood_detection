# -*- coding: utf-8 -*-
"""
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


Visualisation: of impact of no. target pixels on accuracy (Fig.10)

"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from two_stage_detector import TwoStageMatchedFilter
from ds_load import load_ds, load_exbl
from results_util import comp_pr

# ratios of class vectors in the image
RATIOS_V = [0.025, 0.05, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# values of N parameter experessed as a ratio of class vectors
RATIOS_N = [0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1000]


def prepare_no_blood(name_scene="F(1)", id_exbl=24, index=0):
    """
    performs the experiment from discussion:
    number of blood pixels
    data for Fig.10
    parameters:
        name_scene: name of the output image
        id_exbl: id of the reference spectrum
        index: iteration of the experiment
    """
    data_scene, anno_scene = load_ds(name_scene)

    blood_exbl = load_exbl(id_exbl)

    X_bg = data_scene[anno_scene != 1]
    X_blood_raw = data_scene[anno_scene == 1]

    N_blood = len(X_blood_raw)

    ratio_V = RATIOS_V
    ratio_N = RATIOS_N

    res = np.zeros((len(ratio_V), len(ratio_N)))

    indices = np.arange(len(X_blood_raw))
    np.random.shuffle(indices)

    iii = 0
    for i_v, ratio_vectors in enumerate(ratio_V):
        for i_n, t_N in enumerate(ratio_N):
            iii += 1
            print(iii, ratio_vectors, t_N)
            no_vectors = int(ratio_vectors * N_blood)
            N = int(t_N * no_vectors) if t_N <= 1 else int(t_N)

            X_blood = X_blood_raw[indices]
            X_blood = X_blood[:no_vectors]

            X = np.vstack((X_bg, X_blood))
            y = np.concatenate(
                (
                    np.zeros(len(X_bg), dtype=np.int32),
                    np.ones(len(X_blood), dtype=np.int32),
                )
            )
            blood = blood_exbl

            mf = TwoStageMatchedFilter()
            mf.fit(blood, X, N=N, N_supression=0)
            pred_2 = mf.predict(X, stage="second")
            vauc2, _, _ = comp_pr(pred_2, y)
            res[i_v, i_n] = vauc2
    np.savez_compressed("res/exp_no_blood_{}_{}".format(name_scene, index), res=res)


def show_no_blood(name_scene):
    """
    Plots visusalisation of the impact of no. pixels of detection AUC
    Fig.10
    parameters:
        name_scene: name of the output image
    """
    _, anno_scene = load_ds(name_scene)
    N_blood = np.count_nonzero(anno_scene == 1)
    markers = ["s", "v", "<", ">", "1", ".", "p"]
    ratio_V = np.array(RATIOS_V)
    no_vectors = ratio_V * N_blood
    # ratio_N=RATIOS_N
    res = []

    for i in range(1):
        name = "res/exp_no_blood_{}_{}.npz".format(name_scene, i)
        rr = np.load(name)["res"]
        res.append(rr)
    std = np.std(res, axis=0)
    res = np.mean(res, axis=0)

    # print (res.shape,no_vectors.shape)
    # sys.exit()

    plt.rcParams.update({"font.size": 14})
    labels = ["5%", "10%", "25%", "50%", "75%", "100%", "1000"]
    for ii, i_n in enumerate([2, 3, 6]):
        plt.plot(
            no_vectors,
            res[:, i_n],
            label="N={}".format(labels[i_n]),
            marker=markers[ii],
            markevery=2,
        )
    plt.xlabel("No. target pixels in the image")
    plt.ylabel("AUC(PR)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    for i in range(10):
        prepare_no_blood(index=i)
    show_no_blood("F(1)")
