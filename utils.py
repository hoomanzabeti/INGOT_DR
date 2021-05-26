import numpy as np
import os
import datetime
import yaml
import sys
from shutil import copyfile
import inspect
import pandas as pd
from sklearn.metrics import confusion_matrix


def config_decoder(config_inpt):
    """
    Helper Function to parse the config file.
    Based on the stackoverflow post for the return part:
    https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    Parameters:
        config_inpt (dict): dictionary/sub-dictionary from a config file.
    Returns:
        List of all combinations available in config_inpt.
    """
    for keys, vals in config_inpt.items():
        if isinstance(vals, dict):
            # print(vals)
            if 'mode' in config_inpt[keys].keys():
                if config_inpt[keys]['mode'] == 'range':
                    config_inpt[keys] = np.arange(*config_inpt[keys]['values'])
                elif config_inpt[keys]['mode'] == 'list':
                    config_inpt[keys] = config_inpt[keys]['values']
            else:
                config_decoder(config_inpt[keys])
        else:
            config_inpt[keys] = vals
    return config_inpt


def config_reader(config_file_name):
    """
    Main function to parse config file.
    Parameters:
        config_file_name(path): Path to the config file.
    Returns:
        A dictionary of parameters for design and decoding modules.
    """
    try:
        # Read config file
        with open(config_file_name, 'r') as config_file:
            config_dict = yaml.load(config_file, Loader=yaml.FullLoader)
        return config_decoder(config_dict)

    except:
        e = sys.exc_info()[0]
        sys.exit(e)


def path_generator(file_name, file_format, dir_name=None):
    """
    Function to create Results directory and its sub-directories and return saving file path
    Parameters:
        file_name (str): Name of the file.
        file_format (str): File extension.
        dir_name (str): sub-directory name.
    Returns:
        File path
    """
    currentDate = datetime.datetime.now()
    if dir_name is None:
        dir_name = currentDate.strftime("%b_%d_%Y_%H_%M")
    local_path = "./Results/{}".format(dir_name)
    path = os.getcwd()
    if not os.path.isdir(local_path):
        try:
            os.makedirs(path + local_path[1:])
        except OSError:
            print("Creation of the directory %s failed" % path + local_path[1:])
        else:
            print("Successfully created the directory %s" % path + local_path[1:])
    return path + local_path[1:] + "/{}.{}".format(file_name, file_format)


def report_file_path(report_path, report_label, report_extension):
    """
    Function to create a path for report files.
    Parameters:
        report_path (path): Report file path
        report_label (str): Report file label
        report_extension (str): Report file extension. e.g. txt, csv
    Return:
        report_path: Path to the report file.
    """
    report_path = report_path + '/{}.{}'.format(report_label, report_extension)
    return report_path


def dict_key_checker(current_dict, current_key):
    """
    Function to check if a dictionary contains a key.
    Parameters:
        current_dict (dict): The dictionary.
        current_key (str): They key.
    Returns:
        True if the dictionary contains the key. False otherwise.
    """
    if current_key in current_dict.keys():
        return True
    else:
        return False


def result_path_generator(args):
    """
    Function to create a result directory.
    Parameters:
        args (Namespace): Namespace object contains args from command-line including output directory.
    Returns:
        current_path: Current path
        result_path: Path of the result directory. If output directory is not specified output directory would be Results
        with timestamped inner subdirectory.
    """
    current_path = os.getcwd()
    currentDate = datetime.datetime.now()
    if args.output_path is None:
        dir_name = currentDate.strftime("%b_%d_%Y_%H_%M_%S")
        result_path = os.path.join(current_path, "Results/{}".format(dir_name))
    else:
        dir_name = args.output_path
        result_path = os.path.join(current_path, dir_name)
    if not os.path.isdir(result_path):
        try:
            os.makedirs(result_path)
        except OSError:
            print("Creation of the directory %s failed" % result_path)
        else:
            print("Successfully created the directory %s " % result_path)
    # Copy config file
    if os.path.isfile(args.config):
        copyfile(args.config, os.path.join(result_path, 'config.yml'))
    return current_path, result_path


def inner_path_generator(inner_dir, current_path=os.getcwd()):
    """
    Function to create inner subdirectories for intermediate files like design matrix.
    Parameters:
        current_path (path): Current path.
        inner_dir (str): Inner subdirectory name.
    Returns:
        inner_path (path): Inner subdirectory path.
    """
    inner_path = os.path.join(current_path, inner_dir)
    if not os.path.isdir(inner_path):
        try:
            os.makedirs(inner_path)
        except OSError:
            print("Creation of the directory %s failed" % inner_path)
        else:
            print("Successfully created the directory %s " % inner_path)
    return inner_path


def param_distributor(param_dictionary, function_name):
    """
    Function to distribute parameters obtained form the config file to the corresponding modules/functions.
    Parameters:
        param_dictionary (dict): Dictionary of parameters.
        function_name (str): Name of the corresponding function.
    Returns:
        passing_param (dict): Dictionary of parameter which is passing to the function.
        remaining_param (dict): Dictionary of parameter which is NOT passing to the function.
    """
    passing_param = {k: param_dictionary[k] for k in inspect.signature(function_name).parameters if
                     k in param_dictionary}
    remaining_param = {k: inspect.signature(function_name).parameters[k].default if
    inspect.signature(function_name).parameters[k].default != inspect._empty else None for k in
                       inspect.signature(function_name).parameters if k not in passing_param}
    return passing_param, remaining_param


def shap_vals(model, X_train, X_test, shap_kernel):
    """
    Returns the SHAP values of the model

    Parameters:
        model (obj): Model's object after training.
        X_train (DataFrame,2d-array): Training data dataframe/2d-array.
        X_test (DataFrame,2d-array): Testing data dataframe/2d-array.
        shap_kernel (str): SHAP kernel.
    Returns:
        feature_importance (DataFrame): Sorted dataframe of features and their SHAP values.
    """
    import shap
    if shap_kernel == 'TreeExplainer':
        shap_values = shap.TreeExplainer(model).shap_values(X_test)
        vals = np.abs(shap_values[1]).mean(0)
    elif shap_kernel == 'LinearExplainer':
        shap_values = shap.LinearExplainer(model, X_train).shap_values(X_test)
        vals = np.abs(shap_values).mean(0)
    elif shap_kernel == 'KernelExplainer':
        X_train_summary = shap.kmeans(X_test, 50)
        shap_values = shap.KernelExplainer(model.predict_proba, X_train_summary, link="logit").shap_values(X_test,
                                                                                                           nsamples=100)
        vals = np.abs(shap_values[1]).mean(0)
    feature_importance = pd.DataFrame(list(zip(X_train.columns, vals)),
                                      columns=['col_name', 'feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    return feature_importance


def tn(y_true, y_pred):
    """
    Returns number of true negatives.

    Parameters:
        y_true (narray): Numpy array of ture labels.
        y_pred (narray): Numpy array of predicted labels.
    Return:
        Number of true negatives.
    """
    return confusion_matrix(y_true, y_pred)[0, 0]


def fp(y_true, y_pred):
    """
    Returns number of false positives.

    Parameters:
        y_true (narray): Numpy array of ture labels.
        y_pred (narray): Numpy array of predicted labels.
    Return:
        Number of false positives.
    """
    return confusion_matrix(y_true, y_pred)[0, 1]


def fn(y_true, y_pred):
    """
    Returns number of false negatives.

    Parameters:
        y_true (narray): Numpy array of ture labels.
        y_pred (narray): Numpy array of predicted labels.
    Return:
        Number of false negatives.
    """
    return confusion_matrix(y_true, y_pred)[1, 0]


def tp(y_true, y_pred):
    """
    Returns number of true positives.

    Parameters:
        y_true (narray): Numpy array of ture labels.
        y_pred (narray): Numpy array of predicted labels.
    Return:
        Number of true positives.
    """
    return confusion_matrix(y_true, y_pred)[1, 1]


def TNR(y_true, y_pred):
    """
        Returns True Negative Rate.

        Parameters:
            y_true (narray): Numpy array of ture labels.
            y_pred (narray): Numpy array of predicted labels.
        Return:
            True Negative Rate.
    """
    return confusion_matrix(y_true, y_pred)[0, 0] / (confusion_matrix(y_true, y_pred)[0, 0] +
                                                     confusion_matrix(y_true, y_pred)[0, 1])


def my_import(name):
    """
    *** from the python documentation  ***
    Import a module when with submodules split by dot.

    Parameters:
        name of the module
    Return:
        Loaded module
    """
    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def abs_path(file_path):
    return os.path.join(os.getcwd(), file_path)


if __name__ == '__main__':
    """
    Main method for testing config_reader
    """
    print(config_reader('machineLearningComparison/config.yml'))
