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

HyperBlood utility API:
    * Loader for dataset files
    * Utility functions (e.g. simplified RGB visualisation)

Warning:
    * By default, data is cleared by removing noisy bands and broken line in the image. 
    * Note that the 'F(2k)' image was captured with different camera. Its bands were interpolated 
    to match remaining images. However, due to spectral range differences between cameras, it has 
    less bands. After cleaning (default) all images have the same matching 113 bands.

Spectral library of blood samples
    * Note that the code expects a spectral library of blood samples. This library is not part of the dataset
     (external resource, see the preprint) and must be obtained separately. If such library is not provided, 
     the code will substitute blood spectra with mean spectrum from F(1) in get_exbl_spectrum()

NOISY_BANDS_INDICES = np.array([0,1,2,3,4,48,49,50,121,122,123,124,125,126,127])
"""

import unittest
import spectral.io.envi as envi
import numpy as np
import matplotlib.pyplot as plt
from saveload import bz2load
from scipy.interpolate import interp1d

# image codes
IMAGES = [
    "A(1)",
    "B(1)",
    "C(1)",
    "D(1)",
    "E(1)",
    "E(7)",
    "E(21)",
    "F(1)",
    "F(1a)",
    "F(1s)",
    "F(2)",
    "F(2k)",
    "F(7)",
    "F(21)",
]

# wavelenths of the camera
WAVELENGTHS_SOC = np.array(
    [
        376.8200,
        381.7583,
        386.7018,
        391.6505,
        396.6044,
        401.5636,
        406.5280,
        411.4977,
        416.4725,
        421.4525,
        426.4379,
        431.4284,
        436.4241,
        441.4251,
        446.4313,
        451.4427,
        456.4594,
        461.4813,
        466.5084,
        471.5408,
        476.5783,
        481.6211,
        486.6691,
        491.7224,
        496.7808,
        501.8445,
        506.9134,
        511.9876,
        517.0670,
        522.1515,
        527.2413,
        532.3364,
        537.4367,
        542.5422,
        547.6529,
        552.7689,
        557.8901,
        563.0164,
        568.1481,
        573.2849,
        578.4270,
        583.5743,
        588.7269,
        593.8846,
        599.0476,
        604.2158,
        609.3892,
        614.5679,
        619.7518,
        624.9409,
        630.1353,
        635.3348,
        640.5396,
        645.7496,
        650.9649,
        656.1853,
        661.4111,
        666.6420,
        671.8782,
        677.1195,
        682.3661,
        687.6179,
        692.8750,
        698.1372,
        703.4047,
        708.6775,
        713.9554,
        719.2386,
        724.5271,
        729.8207,
        735.1196,
        740.4236,
        745.7329,
        751.0475,
        756.3672,
        761.6923,
        767.0225,
        772.3580,
        777.6987,
        783.0445,
        788.3956,
        793.7520,
        799.1135,
        804.4803,
        809.8524,
        815.2296,
        820.6121,
        825.9998,
        831.3927,
        836.7909,
        842.1942,
        847.6028,
        853.0167,
        858.4358,
        863.8600,
        869.2896,
        874.7243,
        880.1642,
        885.6095,
        891.0598,
        896.5155,
        901.9764,
        907.4425,
        912.9138,
        918.3903,
        923.8721,
        929.3591,
        934.8514,
        940.3488,
        945.8514,
        951.3594,
        956.8725,
        962.3909,
        967.9144,
        973.4432,
        978.9773,
        984.5165,
        990.0610,
        995.6107,
        1001.1656,
        1006.7258,
        1012.2913,
        1017.8618,
        1023.4377,
        1029.0188,
        1034.6050,
        1040.1965,
        1045.7932,
    ]
)

# change this to your DS location
PATH_DATA = "HyperBlood/"
PATH_EXBL = "/"
FILE_EXBL = "blood_raw.bz2"

# images used in the study
IMAGES = [
    {"name_frame": "F(1a)", "name_scene": "F(1)", "id_exbl": 24, "code": "F(1)"},
    {"name_frame": "F(1a)", "name_scene": "F(1s)", "id_exbl": 21, "code": "F(1s)"},
    {"name_frame": "F(1)", "name_scene": "F(1a)", "id_exbl": 20, "code": "F(1a)"},
    {"name_frame": "F(2k)", "name_scene": "F(2)", "id_exbl": 15, "code": "F(2)"},
    {"name_frame": "F(2)", "name_scene": "F(2k)", "id_exbl": 15, "code": "F(2k)"},
    {"name_frame": "F(2)", "name_scene": "F(7)", "id_exbl": 12, "code": "F(7)"},
    {"name_frame": "F(7)", "name_scene": "F(21)", "id_exbl": 5, "code": "F(21)"},
    {"name_frame": "F(1)", "name_scene": "D(1)", "id_exbl": 22, "code": "D(1)"},
    {"name_frame": "F(1)", "name_scene": "A(1)", "id_exbl": 21, "code": "A(1)"},
    {"name_frame": "F(1a)", "name_scene": "B(1)", "id_exbl": 20, "code": "B(1)"},
    {"name_frame": "F(1)", "name_scene": "C(1)", "id_exbl": 21, "code": "C(1)"},
    {"name_frame": "F(1a)", "name_scene": "E(1)", "id_exbl": 20, "code": "E(1)"},
    {"name_frame": "F(7)", "name_scene": "E(7)", "id_exbl": 12, "code": "E(7)"},
    {"name_frame": "F(21)", "name_scene": "E(21)", "id_exbl": 5, "code": "E(21)"},
]


# ------------------------ DATA LOADING ------------------------------------


def load_ds(name, normalise=True, remove_uncertain_blood=False):
    """
    Prepares data for experiment: get_data/get_anno+normalisation
    Parameters:
    ---------------------
    name: image name
    normalise: if True, performs median spectra normalisation
                warning: the data is no longer reflectance
    remove_uncertain_blood: if True, delete class 8

    Returns:
    -----------------------
    datacube, GT (2D array)

    """
    data, _ = get_data(name)
    anno = get_anno(name, remove_uncertain_blood=remove_uncertain_blood)
    if normalise:
        X = data.reshape(-1, data.shape[2])
        X = spectra_normalisation(X)
        data = X.reshape(data.shape)
    return data, anno


def get_data(name, remove_bands=True, clean=True):
    """
    Returns HSI data from a datacube

    Parameters:
    ---------------------
    name: image name
    remove_bands: if True, noisy bands are removed (leaving 113 bands)
    clean: if True, remove damaged line

    Returns:
    -----------------------
    data, wavelenghts as numpy arrays (float32)
    """
    name = convert_name(name)
    filename = "{}data/{}".format(PATH_DATA, name)
    hsimage = envi.open("{}.hdr".format(filename), "{}.float".format(filename))
    wavs = np.asarray(hsimage.bands.centers)
    data = np.asarray(hsimage[:, :, :], dtype=np.float32)

    # removal of damaged sensor line
    if clean and name != "F_2k":
        data = np.delete(data, 445, 0)

    if not remove_bands:
        return data, wavs
    return data[:, :, get_good_indices(name)], wavs[get_good_indices(name)]


def get_anno(name, remove_uncertain_blood=True, clean=True):
    """
    Returns annotation (GT) for data files as 2D int numpy array
    Classes:
    0 - background
    1 - blood
    2 - ketchup
    3 - artificial blood
    4 - beetroot juice
    5 - poster paint
    6 - tomato concentrate
    7 - acrtylic paint
    8 - uncertain blood

    Parameters:
    ---------------------
    name: name
    clean: if True, remove damaged line
    remove_uncertain_blood: if True, removes class 8

    Returns:
    -----------------------
    annotation as numpy 2D array
    """
    name = convert_name(name)
    filename = "{}anno/{}".format(PATH_DATA, name)
    anno = np.load(filename + ".npz")["gt"]
    # removal of damaged sensor line
    if clean and name != "F_2k":
        anno = np.delete(anno, 445, 0)
    # remove uncertain blood + technical classes
    if remove_uncertain_blood:
        anno[anno > 7] = 0
    else:
        anno[anno > 8] = 0

    return anno


# ------------------------ UTILITY ------------------------------------


def spectra_normalisation(X, mode="median"):
    """
    Performs spectra normalisation, dividing each spectrum by their median/mean
    the intention is to standardize the lighting between pixels

    Parameters:
    X: 2D input array
    mode: median, mean or max (normalisation)

    Returns:
    A normalised copy of the input array
    """
    X2 = X.copy()
    norms = None
    if mode == "median":
        norms = np.median(X2, axis=1)
    elif mode == "mean":
        norms = np.mean(X2, axis=1)
    elif mode == "max":
        norms = np.max(X2, axis=1)
    else:
        raise NotImplementedError
    for i in range(len(norms)):
        X2[i] = X2[i] / norms[i]
    return X2


def get_good_indices(name=None):
    """
    Returns indices of bands which are not noisy

    Parameters:
    ---------------------
    name: name
    Returns:
    -----------------------
    numpy array of good indices
    """
    name = convert_name(name) if name is not None else ""
    if name != "F_2k":
        indices = np.arange(128)
        indices = indices[5:-7]
    else:
        indices = np.arange(116)
    indices = np.delete(indices, [43, 44, 45])
    return indices


def convert_name(name):
    """
    Ensures that the name is in the filename format
    Parameters:
    ---------------------
    name: name

    Returns:
    -----------------------
    cleaned name
    """
    name = name.replace("(", "_")
    name = name.replace(")", "")
    return name


def get_rgb(data, wavelengths, gamma=0.7, vnir_bands=[600, 550, 450]):
    """
    Returns an (over)simplified RGB visualization of HSI data

    Parameters:
    ---------------------
    data: data cube as nparray
    annotation: wavelengths - band wavelenghts
    gamma: gamma correction value
    vnir_bands: bands used for RGB

    Returns:
    -----------------------
    rgb image as numpy array
    """
    assert data.shape[2] == len(wavelengths)
    max_data = np.max(data)
    rgb_i = [np.argmin(np.abs(wavelengths - b)) for b in vnir_bands]
    ret = data[:, :, rgb_i].copy() / max_data

    if gamma != 1.0:
        for i in range(3):
            ret[:, :, i] = np.power(ret[:, :, i], gamma)

    return ret


def get_Xy(data, anno):
    """
    return data as 2D arrays (useful e.g. for applying sklearn functions)

    Parameters:
    ---------------------
    data: data cube as nparray
    annotation: 2d annotation array

    Returns:
    -----------------------
    X: 2d array (no. pixels x no.bands)
    y: labels for pixels
    """
    X = data.reshape(-1, data.shape[2])
    y = anno.reshape(-1)
    return X, y


def get_wavelengths():
    """
    returns HSI camera (SOC710) wavelenghts
    """
    indices = get_good_indices()
    return WAVELENGTHS_SOC[indices]


# ------------------------ External blood library ------------------------------------


def load_exbl(sid):
    """
    Returns normalised reference spectra from the spectral library

    Parameters:
    ---------------------
    sid: spectrum id
    Returns:
    -----------------------
    normalised blood spectum
    """
    s_m = get_exbl_spectrum(sid)
    return np.asarray(s_m / np.median(s_m))


def get_exbl_spectrum(sid, cut=True):
    """
    Returns reference spectra from the spectral library, from:
    Majda, Alicja, et al.
    "Hyperspectral imaging and multivariate analysis in the dried blood spots investigations."
    Applied Physics A 124.4 (2018): 312.

    performs interpolation to SOC710 wavelenghts

    Parameters:
    ---------------------
    sid: spectrum id
    cut: if True, a subset of bands is removed to match bands with the rest of (cleaned) images
    Returns:
    -----------------------
    blood spectum (interpolated to SOC710 bands)
    """
    wavelengths = WAVELENGTHS_SOC.copy()[5:-7]
    ss = None
    try:
        mdic_raw = bz2load("{}{}".format(PATH_EXBL, FILE_EXBL))
        w = mdic_raw["wavelenghts"]
        s = mdic_raw[sid]
        f = interp1d(w, s)
        ss = f(wavelengths)
        if cut:
            indices = np.arange(len(ss))
            indices = np.delete(indices, [43, 44, 45])
            ss = ss[indices]
    except:
        print("No blood library file, loading mean spectrum from F(1)")
        data, gt = load_ds("F(1)", normalise=False, remove_uncertain_blood=False)
        ss = np.mean(data[gt == 1, :], axis=0)
    return ss


class LoadTest(unittest.TestCase):
    def test_load(self):
        """
        test image loading
        """
        s = load_exbl(1)
        plt.plot(get_wavelengths(), s)
        plt.ylabel("Reflectance")
        plt.xlabel("Wavelengths")
        plt.show()
        plt.close()

        for im in IMAGES:
            name = im["name_scene"]
            data, wavelengths = get_data(name, remove_bands=True)
            anno = get_anno(name)
            self.assertEqual(data.shape[2], 113)
            self.assertEqual(data.shape[2], wavelengths.shape[0])
            rgb = get_rgb(data, wavelengths)
            plt.subplot(1, 2, 1)
            plt.imshow(rgb, interpolation="nearest")
            plt.subplot(1, 2, 2)
            plt.imshow(anno, interpolation="nearest")
            plt.show()
            plt.close()

    def test_indices(self):
        """
        Ensure F_2k is loaded correctly
        """
        _, wavs = get_data("F_2k", remove_bands=False)
        assert 619.7518 in wavs
        _, wavs = get_data("F_2k", remove_bands=True)
        assert 619.7518 not in wavs
        _, wavs2 = get_data("F_1", remove_bands=True)
        assert np.sum(wavs - wavs2) == 0

        data, wavelengths = get_data("F_1", remove_bands=False)
        self.assertEqual(data.shape[2], 128)
        self.assertEqual(data.shape[2], wavelengths.shape[0])
        data, wavelengths = get_data("F_2k", remove_bands=False)
        self.assertEqual(data.shape[2], 116)
        self.assertEqual(data.shape[2], wavelengths.shape[0])
        anno = get_anno("F_1")
        self.assertEqual(np.count_nonzero(anno == 1), 29598)


if __name__ == "__main__":
    unittest.main()
