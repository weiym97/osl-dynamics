#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Dual Regression for Midnight Scan Club:
# Yiming Wei adapted from Rezvan's codd 2024.

###############################################################################

import os
import numpy as np
import nibabel
import sys

def demean_mat(X, dim):
    if dim == 0:
        Xdem = X - np.tile(np.mean(X, dim)[np.newaxis, :], (X.shape[0], 1))
    else:
        Xdem = X - np.tile(np.mean(X, dim)[:, np.newaxis], (1, X.shape[1]))
    return Xdem


def varnorm_mat(X, dim):
    if dim == 0:
        Xdem = X / np.tile(np.std(X, dim)[np.newaxis, :], (X.shape[0], 1))
    else:
        Xdem = X / np.tile(np.std(X, dim)[:, np.newaxis], (1, X.shape[1]))
    return Xdem

def runDR_v2(D, Pg, desnorm=True):
    # Runs dual regression on the data set to give a set of subject specific maps and time courses
    nv = Pg.shape[0]  # voxles
    nm = Pg.shape[1]  # modes
    nt = D.shape[1]  # time

    nComps = Pg.shape[1]
    EPS_val = 1.0e-5

    # Remove any almost zero maps to keep things more computationally stable
    # Pg[:, np.std(Pg,0) < EPS_val * max(np.std(Pg,0))] = 0.0

    # Regression for the time courses
    iPg = np.linalg.pinv(demean_mat(Pg, 0))
    Dsr = demean_mat(D, 0)
    A = iPg @ Dsr
    # sA = std(A(:,:,s,r)');
    # interimA=A(:,:,s,r);
    # interimA(sA < EPS_val * max(sA), :) = 0.0;

    # And for the subject maps
    Ds = D.T.copy()
    As = A.T.copy()
    Ds = demean_mat(Ds, 0)
    As = demean_mat(As, 0)
    if desnorm:
        As_norm = varnorm_mat(As, 0)
    else:
        As_norm = As.copy()
    P = np.linalg.pinv(As_norm) @ Ds

    # % Remove any almost zero maps
    # sP = std(P(:,:,s));
    # interimP=P(:,:,s);
    # interimP(:, sP < EPS_val * max(sP)) = 0.0;
    # P(:,:,s)=interimP;
    return P.T, A

if __name__ == '__main__':
    print("-" * 80)
    print("running dual regression...")
    print()
    group_ica_path = str(sys.argv[1])
    raw_file_list_dir = str(sys.argv[2])
    outpath = str(sys.argv[3])
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    gICA_img = nibabel.load(group_ica_path)
    gICA = gICA_img.get_fdata().T

    # Initialise subj_run_names and dataloc as list
    subj_run_names = []
    dataloc = []

    # Read all lines from the text file into a list
    with open(raw_file_list_dir, 'r') as file:
        lines = file.readlines()

    # Process each line to extract the subject name from the last part
    for line in lines:
        line = line.strip()  # Remove leading/trailing whitespace
        line_parts = line.split('/')  # Split the line by '/'
        filename_parts = line_parts[-1].split('_')  # Split the last part by '_'
        subject_name = '_'.join(filename_parts[:2])  # Select the first two parts and join with '_'

        subj_run_names.append(subject_name)
        dataloc.append(line)


    for subname, subpath in zip(subj_run_names, dataloc):
        print(subname)
        data_img = nibabel.load(subpath)
        data = data_img.get_fdata().T

        Ps, As = runDR_v2(data, gICA, desnorm=True)
        outname = os.path.join(outpath, f'{subname}_smaps.npy')
        np.save(outname, Ps)
        outname = os.path.join(outpath, f'{subname}_timecourses.npy')
        np.save(outname, As.T)


