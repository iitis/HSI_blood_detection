# -*- coding: utf-8 -*-
"""
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

Visualisation: comparison of spectra in the 1st and 2nd stage of the algorithm (Fig.9)
"""

import numpy as np
import matplotlib.pyplot as plt
from two_stage_detector import TwoStageMatchedFilter
from os.path import isfile
from ds_load import load_ds, load_exbl, get_wavelengths
from _ctypes import ArgumentError
from experiment_detection import decode_N


def show_means_compared(name_frame, name_scene, id_exbl, N=1000):
    """
    Plots the visualisation of impact of the second stage of the algorithm on
    spectra, Fig.9
    parameters:
        name_frame: name of the source image (inductive scenario)
        name_scene: name of the output image
        id_exbl: id of the reference spectrum
        N: no. vectors in the 2nd stage as int or percentage string e.g. '33p'
    """

    data_frame, anno_frame = load_ds(name_frame, normalise=False)
    data_scene, anno_scene = load_ds(name_scene, normalise=False)
    blood_exbl = load_exbl(id_exbl)

    blood_frame = np.mean(data_frame[anno_frame == 1], axis=0)
    blood_scene = np.mean(data_scene[anno_scene == 1], axis=0)

    assert len(blood_exbl) == len(blood_frame)
    assert len(blood_exbl) == len(blood_scene)

    markers = ["s", "v", "<", ">", "1", ".", "p"]

    n_blood = np.count_nonzero(anno_scene == 1)
    N_v = decode_N(N, n_blood)

    wav = get_wavelengths()

    X_data = data_scene.reshape(-1, data_scene.shape[2]).copy()
    mf = TwoStageMatchedFilter()

    mf.fit(blood_exbl, X_data, N=N_v, N_supression=0)

    plt.rcParams.update({"font.size": 14})
    plt.plot(wav, blood_scene, label="Image mean", marker=markers[0], markevery=0.3)
    plt.plot(wav, blood_exbl, label="Library", marker=markers[1], markevery=0.3)
    plt.plot(wav, mf.mf_2.mu_t, label="Second stage", marker=markers[2], markevery=0.3)

    plt.ylabel("Reflectance")
    plt.xlabel("Wavelenghts")

    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def compare_means():
    """
    plots Fig.9
    """
    IMAGES = [
        {"name_frame": "F(1a)", "name_scene": "F(1)", "id_exbl": 24, "code": "F(1)"}
    ]
    for im in IMAGES:
        show_means_compared(im["name_frame"], im["name_scene"], im["id_exbl"], N=1000)


if __name__ == "__main__":
    compare_means()
