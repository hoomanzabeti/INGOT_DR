'''
The following code was copied from a stackoverflow post:
https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
'''
import re
import numpy as np
import itertools
import os
import datetime

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]

def config_decoder(config_inpt):
    '''
    TODO: I used the following stackoverflow post for the return part:
    https://stackoverflow.com/questions/5228158/cartesian-product-of-a-dictionary-of-lists
    '''
    for keys, vals in config_inpt.items():
        if isinstance(vals, dict):
            print(vals)
            if config_inpt[keys]['mode'] == 'range':
                config_inpt[keys] = np.arange(*config_inpt[keys]['values'])
            elif config_inpt[keys]['mode'] == 'scalar':
                config_inpt[keys] = config_inpt[keys]['values']
            elif config_inpt[keys]['mode'] == 'exact':
                config_inpt[keys].pop('mode')
                config_inpt[keys] = [config_inpt[keys]]
        else:
            config_inpt[keys] = [vals]
    return [dict(zip(config_inpt.keys(), vals)) for vals in itertools.product(*config_inpt.values())]

def path_generator(file_path, file_name, file_format):
    currentDate = datetime.datetime.now()
    # TODO: change dir_name to unique code if you want to run multiple configs
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