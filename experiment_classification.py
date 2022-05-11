# -*- coding: utf-8 -*-
"""
************************************************************************
Copyright 2020 Institute of Theoretical and Applied Informatics, 
Polish Academy of Sciences (ITAI PAS) https://www.iitis.pl
author: M. Romasewszki, mromaszewski@iitis.pl

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

Code for the classification experiment (Sec. 6.2 in the discussion)
"""
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score

from ds_load import load_ds, get_Xy
from sklearn.preprocessing import scale


def getSVM(X, y, random_state=0):
    """
    Trains SVM with stratified CV
    parameters:
        X: 2D array of examples
        y: labels
        random_state: experiments random state
    returns:
        trained classifier
    """
    gamma = 1 / (X.shape[1] * X.var())
    gammas = [gamma * (10**p) for p in [-2, -1, 0, 1, 2]]

    parameters = {"C": [(10**p) for p in [-2, -1, 0, 1, 2]], "gamma": gammas}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        cv=skf, estimator=SVC(probability=False, kernel="rbf"), param_grid=parameters
    )
    gs.fit(X, y)
    return gs.best_estimator_


def select_subset(y, size=500, min_n_class=5, random_state=0):
    """
    Stratified selection of a random subset from a dataset
    useful for uniform, random data selection

    parameters:
    y: labels
    size: if size>1: no. examples/class, if size<1: ratio of examples/class
    min_n_class: minimum number of examples/class (if exists)
    random_state: if random_state>=0: random state of the rng

    returns:
    subset_indices,remaining indices
    """
    if random_state > 0:
        np.random.seed(random_state)

    indices = np.arange(len(y))
    rets = []
    for u in np.unique(y):
        ind = indices[y == u]
        np.random.shuffle(ind)
        if size >= 1:
            rets.append(ind[:size])
        else:
            N = np.count_nonzero(y == u)
            N = int(N * size)
            if N < min_n_class:
                N = np.min([min_n_class, np.count_nonzero(y == u)])
            rets.append(ind[:N])

    rets = np.ravel(np.concatenate(rets))
    return rets, np.setdiff1d(indices, rets)


def classification_experiment(name="d01_frame_I300", binary=False):
    """
    Implementation of the experiment in Sec.6.2
    parameters:
        name: name of the HSI image
        binary: if True, the problem is treated as binary (blood/others) for acc
            computation. If False, this is a multi-class classification experiment
    returns:
        trained classifier
    """
    data, gt = load_ds(name, remove_uncertain_blood=True)
    X, y = get_Xy(data, gt)

    X = X[y != 0]
    y = y[y != 0]

    X = scale(X)
    if binary:
        y[y != 1] = 2
    acc = []
    prec = []
    rec = []

    for i in range(10):
        print(i, binary)
        i_train, i_test = select_subset(y, size=0.05, random_state=i)
        svm = getSVM(X[i_train], y[i_train], random_state=i)
        pred = svm.predict(X[i_test])
        acc.append(accuracy_score(y[i_test], pred))
        y2 = y.copy()
        y2[y2 != 1] = 2
        pred[pred != 1] = 2
        prec.append(precision_score(y2[i_test], pred, pos_label=1))
        rec.append(recall_score(y2[i_test], pred, pos_label=1))

    print(name)
    print("acc: {:0.2f}({:0.2f})".format(np.mean(acc) * 100.0, np.std(acc) * 100.0))
    print("prec: {:0.2f}({:0.2f})".format(np.mean(prec) * 100.0, np.std(prec) * 100.0))
    print("rec: {:0.2f}({:0.2f})".format(np.mean(rec) * 100.0, np.std(rec) * 100.0))


if __name__ == "__main__":
    classification_experiment(name="E(1)", binary=True)
