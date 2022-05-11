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

Detection performance measures (AUC+ROC/PR)
"""

import unittest
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve


def comp_pr(res, gt):
    """
    computers Precision-Recall measures

    parameters:
        res: detection result
        gt: ground truth (class 1 is the target)
    returns:
        AUC, precision, recall
    """
    y_pred = np.ravel(res.copy())
    y_test = np.ravel(gt.copy())
    y_test[y_test != 1] = 0
    precision, recall, _ = precision_recall_curve(y_test, y_pred, pos_label=1)
    vauc = auc(recall, precision)
    return vauc, precision, recall


def comp_roc(res, gt):
    """
    computers ROC measures

    parameters:
        res: detection result
        gt: ground truth (class 1 is the target)
    returns:
        AUC, FPR, TPR
    """
    y_pred = np.ravel(res.copy())
    y_test = np.ravel(gt.copy())
    y_test[y_test != 1] = 0
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    vauc = auc(fpr, tpr)
    return vauc, fpr, tpr


class LoadTest(unittest.TestCase):
    def test_random(self):
        N = 10000
        gt = np.random.randint(2, size=N)
        res = np.random.rand(N)
        import matplotlib.pyplot as plt

        vauc_pr, precision, recall = comp_pr(res, gt)
        vauc_roc, fpr, tpr = comp_roc(res, gt)
        self.assertLess(np.abs(vauc_pr - 0.5), 0.05)
        self.assertLess(np.abs(vauc_roc - 0.5), 0.05)

        plt.subplot(1, 2, 1)
        plt.plot(
            recall,
            precision,
            label="PR AUC:{:0.2f}".format(vauc_pr),
            alpha=0.7,
            markevery=0.3,
            lw=2,
        )
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(
            fpr,
            tpr,
            label="ROC AUC:{:0.2f}".format(vauc_roc),
            alpha=0.7,
            markevery=0.3,
            lw=2,
        )
        plt.legend()
        plt.show()
        plt.close()


if __name__ == "__main__":
    unittest.main()
