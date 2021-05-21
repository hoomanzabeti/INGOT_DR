#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 15:06:21 2020

@author: hoomanzabeti
"""

from scipy import sparse
import pandas as pd


def data_loader(isolate_name, snp_loc, feature_matrix, label, drug_name, feature_freq_lower_bound=1):
    # Loading isolate names/IDs
    iso_df = pd.read_csv(isolate_name, usecols=['name/ pos- ref- alt'])

    # Loading SNPs position, reference and alternates                                                                                                
    #snp_df = pd.read_csv(snp_loc, header=None, nrows=1)
    snp_df = pd.read_csv(snp_loc)

    # Loading feature matrix (SNPs presence/absence matrix) and converting it                                                                        
    # to dense matrix                                                                                                                                
    sparse_matrix = sparse.load_npz(feature_matrix)
    matrix_df = pd.DataFrame.sparse.from_spmatrix(sparse_matrix)
    matrix_df = matrix_df.sparse.to_dense().astype('uint8')

    # Loading labels                                                                                                                                 
    label_df = pd.read_csv(label, index_col='id')

    # Concatenating dataframes together
    df_iso = pd.concat([iso_df, matrix_df], axis=1, join='inner')
    df_iso.columns = list(snp_df.iloc[0, :])
    df_iso = df_iso.set_index('name/ pos- ref- alt')
    # Set features frequency lower bound                                                                                                             
    df_iso = df_iso.loc[:, df_iso.sum() >= feature_freq_lower_bound]
    # Concatenating with labels
    complete_matrix = pd.concat([df_iso, label_df], axis=1, join='inner')
    complete_matrix = complete_matrix.loc[complete_matrix[drug_name].notna()]

    # Splitting data to feature matrix X and label vector y for selected drug
    _, n = label_df.shape
    X = complete_matrix.iloc[:, :-n]
    y = complete_matrix[drug_name]

    return X, y
