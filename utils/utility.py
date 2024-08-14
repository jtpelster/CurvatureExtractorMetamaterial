"""
Collection of usefull functions for saving and loading
"""
from pathlib import Path
import json
import codecs
from warnings import warn
import numpy as np
import math
import os
import open3d as otd
import re
from numpy import cos
from numpy import sin
from numpy import pi


def load_json_file(filename):
    with open(filename + ".json", "r", encoding="utf-8") as f:
        json_file = json.load(f)
        f.close()
    return json_file


def get_variables_from_json(filename, variable_name_list):
    """
    Gets variables from a json file.
    Parameters:
        filename (string): filename including path.
        variable_name_list (list of strings): list of variable names to extract.

    Returns:
        variables (list or string): the variables that were selected in a list.
    """
    if isinstance(variable_name_list, str):
        variable_name_list = [variable_name_list]
    variable_dictionary = load_json_file(filename)
    # variable_dictionary = json.loads(json_file)
    variables = []
    for variable_name in variable_name_list:
        variable = variable_dictionary[variable_name]
        variables.append(variable)
    if len(variable_name_list) == 1:
        variables = variables[0]
    return variables


def load_matching_variables(file, regex, matching_idx):
    regex_match = re.compile(regex)
    json_file = load_json_file(str(file))
    filtered_keys = [
        regex_match.match(key)[0]
        for key in json_file
        if not regex_match.match(key) is None
    ]
    variables = [json_file[key] for key in filtered_keys]
    return variables


def find_match_in_variables(file, regex, matching_idx):
    regex_match = re.compile(regex)
    json_file = load_json_file(str(file))
    matches = [
        regex_match.match(key)[matching_idx]
        for key in json_file
        if not regex_match.match(key) is None
    ]
    return matches


class npEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.int32):
            return int(obj)
        return json.JSONEncoder.default(self, obj)


def parameters_to_folder_name(parameters):
    folder_names = [
        "TO" + str(TO) + "_TN" + str(TN) + "_HO" + str(HO) + "_HN" + str(HN)
        for TO, TN, HO, HN in parameters
    ]
    return folder_names


def copy_json(original_file, new_file):
    try:
        original_file_data = load_json_file(original_file)
    except FileNotFoundError as FNFE:
        raise ValueError("The original file does not exist!")

    error_switch = 0
    if not os.path.isdir(os.path.dirname(new_file)):
        os.makedirs(os.path.dirname(new_file))
    with open(new_file + ".json", "w", encoding="utf-8") as f:
        json.dump(original_file_data, f, indent=4, cls=npEncoder)
    pass


def save_json(filename, variable_name_list, variable_list, overwrite):
    """
    Saves variables to json file, converts np arrays to lists.
    Parameters:
        filename (string): filename including path.
        variable_name_list (list of strings): List of variable names in the same
            order as variable list.
        variable_list (list of strings, dicts, lists, or numbers and their
            compositions): List of variables in the same order as variable name list.
    """
    # Convert all arrays to lists
    if isinstance(variable_name_list, str):
        variable_name_list = [variable_name_list]

    processed_list = []
    for variable in variable_list:
        variable = getattr(variable, "tolist", lambda: variable)()
        processed_list.append(variable)
    saving_dict = {}
    error_switch = False
    try:
        original_file = load_json_file(filename)
        print("json file exists:" + filename)
        if overwrite:
            print("Replacing data in file")
        else:
            print("Updating file with new information")
    except FileNotFoundError as FNFE:
        print("json file does not exist:" + filename)
        print("creating new file")
        error_switch = True
    except ValueError as JDE:
        print("Can't read file? Trying to create new file.")
        error_switch = True

    if (not overwrite) and not error_switch:
        saving_dict.update(original_file)
    elif error_switch:
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

    for variable, variable_name in zip(processed_list, variable_name_list):
        saving_dict.update({variable_name: variable})

    error_switch = 0
    with open(filename + ".json", "w", encoding="utf-8") as f:
        try:
            json.dump(saving_dict, f, indent=4, cls=npEncoder)
        except Exception as error:
            warn(str(error))
            warn("Resaving original file before erroring")
            json.dump(original_file, f, indent=4, cls=npEncoder)
            error_switch = 1

    if error_switch:
        raise ValueError("Original File Saved, but new data not, recheck errors")
    pass


def get_files(folder, extension):
    files = list(Path(folder).rglob("*" + extension))
    return files
