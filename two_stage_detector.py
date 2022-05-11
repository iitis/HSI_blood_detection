# -*- coding: utf-8 -*-
"""
************************************************************************
Copyright 2020 Institute of Theoretical and Applied Informatics, 
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
author: M. Romaszewszki, mromaszewski@iitis.pl

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


The two-stage detector used in experiments
"""
import unittest
import numpy as np
import matplotlib.pyplot as plt

from target_detectors import MatchedFilter


class TwoStageMatchedFilter:
    """
    Two stage matched filter

    The first stage applies matched filter to data
    The second stage retrains matched filter using N ,,best'' matches
    from data and potentially supressing best N_supression matches when
    creating second covariance matrix
    """

    def __init__(self):
        self.mf_1 = None
        self.mf_2 = None

    def fit(self, X_target, X_data, N, N_supression=0):
        """
        trains data model

        parameters:
        X_target: 2D array of targets or target mean/spectrum
        X_data: 2D array of backgrounds
        N: the number of vectors used for retraining
        N_supression: the number of vectors supressed when creating
                    a second stage covaraince matrix (consider N_supression=N)

        """
        self.mf_1 = MatchedFilter(X_target=X_target, X_data=X_data)
        pred = self.mf_1.predict(X_data)
        arg = np.argsort(pred)[::-1]
        X_blood = X_data[arg[:N]]
        X_others = X_data if N_supression == 0 else X_data[arg[N_supression:]]
        self.mf_2 = MatchedFilter(X_target=X_blood, X_data=X_others)

    def predict(self, X, stage="second"):
        """
        model-based detection

        parameters:
        X: 2d array of vectors
        stage: "first" for the first stage detector, "second" for the second stage

        returns:
        array of scores (higher is better)
        """
        if stage == "first":
            return self.mf_1.predict(X)
        elif stage == "second":
            return self.mf_2.predict(X)
        else:
            raise NotImplementedError


class Test(unittest.TestCase):
    def test_detector(self):
        np.random.seed(42)
        T = np.random.normal(0, 0.2, (50, 2))
        B = np.random.normal(1, 0.4, (100, 2))
        X = np.vstack((B, T))
        tsmf = TwoStageMatchedFilter()
        for N_supression in [0, 50]:
            tsmf.fit(T, X, N=50, N_supression=N_supression)

            for i_s, stage in enumerate(["first", "second"]):
                plt.subplot(2, 1, 1 + i_s)
                pred = tsmf.predict(X, stage=stage)
                plt.hist(
                    np.ravel(pred[:100]), color="red", label="background", alpha=0.7
                )
                plt.hist(np.ravel(pred[100:]), color="green", label="target", alpha=0.7)
                plt.title(stage)
                plt.legend()
            plt.tight_layout()
            plt.show()
            plt.close()


if __name__ == "__main__":
    unittest.main()
