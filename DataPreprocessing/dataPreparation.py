import os.path
from sklearn.model_selection import train_test_split
from dataLoader import data_loader
import pandas as pd
import csv
import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import utils

def data_preparation(current_drug, current_model, isolate_name_path, snp_loc_path, feature_matrix_path,
                     label_path, output_dir, seed=33, test_size=0.2):
    # Loading  the data for drug 'current_drug'
    snp_matrix_shape_dict = {}
    print_val = lambda p: print('{} {} {}'.format('-' * (50 - len(p) // 2), p, '-' * (50 - len(p) // 2)))
    print_val(current_drug)
    print_val('Data preparation: Start')
    X, y = data_loader(isolate_name=isolate_name_path, snp_loc=snp_loc_path,
                       feature_matrix=feature_matrix_path, label=label_path,
                       drug_name=current_drug, feature_freq_lower_bound=1)

    snp_list = list(X.columns)
    snp_list = [i for i in snp_list if len(i.split(', ')[1]) == 1 and len(i.split(', ')[2]) == 1]
    X = X.filter(items=snp_list)
    print("Feature Matrix shape with duplicates: ", X.shape)
    print("Feature Matrix shape without zeros: ", X.loc[:, (X != 0).any(axis=0)].shape)
    snp_matrix_shape_dict.update(
        {'m': X.shape[0], 'snp': X.shape[1], 'snp_without_zero': X.loc[:, (X != 0).any(axis=0)].shape[1]})
    X = X.T.drop_duplicates().T
    print("Feature Matrix shape without duplicates", X.shape)
    inner_path = utils.inner_path_generator('data', output_dir)
    snp_matrix_shape_dict.update({'m_without_duplicates': X.shape[0], 'snp_without_duplicates': X.shape[1]})
    pd.DataFrame(snp_matrix_shape_dict, index=[0]).to_csv(os.path.join(inner_path, '{}_shape.csv'.format(current_drug)))
    if current_model.lower() != 'kover':
        y.to_frame().to_csv(os.path.join(inner_path, '{}Label.csv'.format(current_drug)))
        X.to_csv(os.path.join(inner_path, 'SNPsMatrix_{}.csv'.format(current_drug)))
    else:
        y.to_csv(os.path.join(inner_path, '{}Metadata.tsv'.format(current_drug)), sep='\t')
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=test_size, stratify=y)
        with open(os.path.join(inner_path, '{}_train_isolates.txt'.format(current_drug)), 'w') as f:
            for item in list(X_train.index):
                f.write("%s\n" % item)
        with open(os.path.join(inner_path, '{}_test_isolates.txt'.format(current_drug)), 'w') as g:
            for item in list(X_test.index):
                g.write("%s\n" % item)
        X = X.T
        X.reset_index(inplace=True)
        X = X.rename(columns={'index': 'kmers'})
        # Name of SNPs should be the same length
        fixed_length = X.kmers.apply(lambda x: len(x)).max()
        X.kmers = X.kmers.apply(lambda x: x + "_" * (fixed_length - len(x)))
        X.to_csv(os.path.join(inner_path, 'SNPsMatrix_{}.tsv'.format(current_drug)), sep='\t', index=False,
                 quoting=csv.QUOTE_NONE)
    print_val('Data preparation: End')


if __name__ == "__main__":
    # Files path
    data_path = lambda d: os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), d)
    isolate_name_path = data_path('data/iso_list.csv')
    snp_loc_path = data_path('data/SNPList.csv')
    feature_matrix_path = data_path('data/sparsetableFeb27.npz')
    label_path = data_path('data/AllLabels.csv')
    # drug_list = ['amikacin', 'capreomycin', 'ciprofloxacin', 'ethambutol', 'ethionamide', 'isoniazid', 'kanamycin',
    #              'moxifloxacin', 'ofloxacin', 'pyrazinamide', 'rifampicin', 'streptomycin']
    drug_list = ['ciprofloxacin']
    output_dir = os.getcwd()
    for drug in drug_list:
        data_preparation(current_drug=drug, current_model='kover', isolate_name_path=isolate_name_path,
                         snp_loc_path=snp_loc_path, feature_matrix_path=feature_matrix_path, label_path=label_path,
                         output_dir=output_dir, seed=33, test_size=0.2)
        data_preparation(current_drug=drug, current_model='INGOT_ML', isolate_name_path=isolate_name_path,
                         snp_loc_path=snp_loc_path, feature_matrix_path=feature_matrix_path, label_path=label_path,
                         output_dir=output_dir, seed=33, test_size=0.2)
