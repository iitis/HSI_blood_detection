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

Matched Filter hyperspectral target detector, based on: 
D. Manolakis at al., "Hyperspectral Image Processing for automatic Target Detection Applications" 
"""
import unittest
import numpy as np
from sklearn.covariance import empirical_covariance
import scipy.linalg as lin
import matplotlib.pyplot as plt


class MatchedFilter:
    """
    Matched Filter detector
    """

    def __init__(self, X_target=None, X_data=None):
        """
        Matched Filter
        by default, fit() is not called

        parameters:
        X_target: None or 2D array of targets or target mean/spectrum
        X_data: None or 2D array of backgrounds

        """
        self.mu_t = None
        self.C_d = None
        self.mu_d = None
        self.d = None
        self.mu = None
        if X_target is not None and X_data is not None:
            self.fit(X_target=X_target, X_data=X_data)

    def fit(self, X_target, X_data):
        """
        trains data model

        parameters:
        X_target: 2D array of targets or target mean/spectrum
        X_data: 2D array of backgrounds

        """
        self.mu_t = np.mean(X_target, axis=0) if len(X_target.shape) > 1 else X_target
        self.C_d = empirical_covariance(X_data)

        self.C_d = lin.inv(self.C_d)
        self.mu_d = np.mean(X_data, axis=0)
        # constant denominator
        self.mu = self.mu_t - self.mu_d
        self.d = np.dot(np.dot(self.mu, self.C_d), self.mu)

    def predict(self, X):
        """
        model-based detection

        parameters:
        X: 2d array of vectors

        returns:
        array of scores (higher is better)
        """
        T = X - self.mu_d
        # old, slow
        # ll = np.array([np.dot(np.dot(t, self.C_d), self.mu) for t in T])

        # new, fast
        ll = np.dot(np.dot(T, self.C_d), self.mu)
        return ll / self.d

    def fit_predict(self, X_target, X_data):
        """
        trains data model and returns scores

        parameters:
        X_target: 2D array of targets or target mean/spectrum
        X_data: 2D array of backgrounds

        returns:
        array of scores (higher is better)
        """
        self.fit(X_target, X_data)
        return self.predict(X_data)


class Test(unittest.TestCase):
    def test_detectors(self):
        np.random.seed(42)
        T = np.random.normal(0, 0.2, (50, 2))
        B = np.random.normal(1, 0.4, (100, 2))
        X = np.vstack((B, T))

        mf = MatchedFilter()
        pred_mf = mf.fit_predict(T, X)

        # one vector used instead of mean
        pred_mf1 = mf.fit_predict(T[0], X)

        names = ["mf", "mf(1)"]
        for i_pred, pred in enumerate([pred_mf, pred_mf1]):
            plt.subplot(len(names), 1, i_pred + 1)
            plt.hist(np.ravel(pred[:100]), color="red", label="background", alpha=0.7)
            plt.hist(np.ravel(pred[100:]), color="green", label="target", alpha=0.7)
            plt.title(names[i_pred])
            plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    def test_detectors_fast(self):
        for rs in np.arange(10) + 42:
            np.random.seed(rs)
            T = np.random.normal(0, 0.2, (50, 2))
            B = np.random.normal(1, 0.4, (100, 2))
            X = np.vstack((B, T))

            mf = MatchedFilter()
            pred_mf = mf.fit_predict(T, X)
            T = X - mf.mu_d
            pred_mf2 = np.array([np.dot(np.dot(t, mf.C_d), mf.mu) for t in T]) / mf.d
            err = np.linalg.norm(pred_mf - pred_mf2)
            self.assertLess(err, 10e-5)


if __name__ == "__main__":
    unittest.main()
