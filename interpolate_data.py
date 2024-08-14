"""
Steps to take:
Provide option to plot specific types of graphs
Either with automatic bounds, or the
maximum and minimum of the whole set provided by the
user for each plot type.
Interpolate current plot type data.
Then find max and min.
Then plot.
"""
import numpy as np
from utils import utility as util
from utils import plot_utilities as putil
import os
import math

MAIN_DATA_FOLDER = "./processed_data/"
MAIN_INTERP_FOLDER = "./interpolated_data/"
DATA_DEFAULT_FOLDER = "/standard_process/"
INTERP_DEFAULT_FOLDER = "/interp_default/"
INTERPOLATION_SIZE_DEFAULT = [500, 500]
BORDER_FRAC_DEFAULT = 0.0
INTERPOLATION_TYPE_DEFAULT = "nearest"
POS_INERPOLATION_TYPE_DEFAULT = "linear"


def save_interpolation(
    file,
    x_grid,
    y_grid,
    data_grid,
    z,
    x_data,
    y_data,
    z_data,
    data_points,
    data_name,
    INTERP_FOLDER,
):
    """
    Wrapper for saving an interpolated file:
    """

    util.save_json(
        INTERP_FOLDER + file + "." + data_name,
        [
            "x_grid",
            "y_grid",
            "z",
            "data_grid",
            "x_data",
            "y_data",
            "z_data",
            "data_points",
        ],
        [x_grid, y_grid, z, data_grid, x_data, y_data, z_data, data_points],
        1,
    )


def interpolate(
    variables,
    interpolation_type,
    pos_interpolation_type,
    interpolation_size,
    border_frac,
):
    # print(variables)
    pos_variables = variables[:, :3]
    # print(pos_variables)
    variables = np.delete(variables, 2, 1)

    parameters = [interpolation_type, interpolation_size, border_frac]
    pos_parameters = [pos_interpolation_type, interpolation_size, border_frac]
    loaded_variables = putil.non_gridded_interpolator(variables, *parameters)
    loaded_pos_variables = putil.non_gridded_interpolator(
        pos_variables, *pos_parameters
    )

    return loaded_variables, loaded_pos_variables


def load_variables(file, variable, independent_variables, independent_file=None):
    if independent_file is None:
        all_vars = util.get_variables_from_json(
            file, [variable] + [independent_variables]
        )
    else:
        dep_var = util.get_variables_from_json(file, variable)
        indep_vars = util.get_variables_from_json(
            independent_file, independent_variables
        )
        all_vars = dep_var + indep_vars
    return all_vars


def process_variables(variables):
    """
    Converts variables into a useable state for interpolation.
    Paramters:
        variables (list of lists): A list with a dependent variable as its
        first entry, and independent variables as its second and/or third.
        If one indep variable, it is assumed that the variables are contained
        within the first two columns in x and y order.
        If two indep variables, then it is assumed that each is an x and y
        array/list respectively.
    Rturns[x,y,z,depnd]
    """
    separate_indep_switch = False
    if len(variables) > 2:
        separate_indep_switch = True
    dep_var = np.reshape(variables.pop(0), [-1, 1])
    dep_var = np.array(dep_var)
    if separate_indep_switch:
        indep_vars = variables.copy()
    else:
        indep_vars = variables[0]
    indep_array_list = []
    if separate_indep_switch:
        for indep_var in indep_vars:
            indep_array_list.append(np.array(indep_var))
        formatted_vars = np.hstack((*indep_array_list, dep_var))
    else:
        formatted_vars = np.hstack((np.array(indep_vars), np.array(dep_var)))
    return formatted_vars


def remove_boundary(
    file, formatted_vars, boundary_type, ignore_boundary,
):
    indexes = util.get_variables_from_json(file, boundary_type)
    if ignore_boundary:
        formatted_vars[indexes, 3:] = float("NaN")

    return formatted_vars


def rotate(data, degrees):
    print(data)
    x, y, z, *data_vals = data.T

    x = x - np.mean(x)
    y = y - np.mean(y)
    print(x)
    print(data_vals)
    angle = degrees * math.pi / 180
    R2 = np.array(
        [[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle),],]
    )

    x1, y1 = R2 @ np.array([x, y])
    print(np.array([x1]))
    rot_data = np.vstack((np.array(x1).T, np.array(y1).T, np.array(z).T, *data_vals))
    print(rot_data)
    return rot_data.T


def interpolate_save(
    file,
    variable,
    independent_variable,
    boundary_def,
    ignore_boundary,
    INTERP_FOLDER,
    DATA_FOLDER,
    interpolation_size,
    border_frac,
    interpolation_type,
    pos_interpolation_type,
    independent_file=None,
    angle=30,
):
    loaded_variables = load_variables(
        DATA_FOLDER + file, variable, independent_variable, independent_file
    )
    processed_variables = process_variables(loaded_variables)
    rotated_variables = rotate(processed_variables, angle)
    chopped_variables = remove_boundary(
        DATA_FOLDER + file, rotated_variables, boundary_def, ignore_boundary,
    )
    interpolated_variable, pos_interp_variable = interpolate(
        chopped_variables,
        interpolation_type,
        pos_interpolation_type,
        interpolation_size,
        border_frac,
    )
    save_interpolation(
        file,
        *interpolated_variable,
        pos_interp_variable[2],
        *chopped_variables.T,
        variable,
        INTERP_FOLDER
    )


def interpolate_fx(
    files_to_process,
    variables_to_process,
    ignore_boundaries,
    INTERP_FOLDER=MAIN_INTERP_FOLDER + INTERP_DEFAULT_FOLDER,
    DATA_FOLDER=MAIN_DATA_FOLDER + DATA_DEFAULT_FOLDER,
    interpolation_size=INTERPOLATION_SIZE_DEFAULT,
    border_frac=BORDER_FRAC_DEFAULT,
    interpolation_type=INTERPOLATION_TYPE_DEFAULT,
    pos_interpolation_type=POS_INERPOLATION_TYPE_DEFAULT,
):
    corresp_independent_variables = {
        "gaussian_curvature": "point_cloud",
        "normalized_gaussian_curvature": "point_cloud",
        "hexagon_gaussian_curvature": "hex_coordinates",
        "mean_curvature": "point_cloud",
        "normalized_mean_curvature": "point_cloud",
        "curv_rmse": "point_cloud",
        "isotropic_gaussian_curvature": "hex_coordinates",
        "normalized_isotropic_gaussian_curvature": "hex_coordinates",
        "isotropic_rmse": "hex_coordinates",
        "isotropic_comparison": "hex_coordinates",
        "normalized_isotropic_comparison": "hex_coordinates",
        "area_expansion": "hex_coordinates",
        "area_dist_error": "hex_coordinates",
        "relative_anisotropy": "hex_coordinates",
        "anisotropy": "hex_coordinates",
        "out_of_plane_aniosotropy_variation": "hex_coordinates",
    }

    boundary_def = {
        "point_cloud": "edge_point_idx",
        "hex_coordinates": "outer_hexagons",
    }

    if not os.path.isdir(INTERP_FOLDER):
        os.makedirs(INTERP_FOLDER)
    for file in files_to_process:
        for variables in variables_to_process:
            interpolate_save(
                file,
                variables,
                corresp_independent_variables[variables],
                boundary_def[corresp_independent_variables[variables]],
                ignore_boundaries[variables],
                INTERP_FOLDER,
                DATA_FOLDER,
                interpolation_size,
                border_frac,
                interpolation_type,
                pos_interpolation_type,
            )

    print("Interpolation Complete")


if __name__ == "__main__":

    
    # files_to_process = [
    #     "EXP_96_1",
    #     "EXP_96_2",
    #     "EXP_96_3",
    #     "EXP_96_4",
    # ]

    # BIG LIST OF VARIABLES
    # "normalized_gaussian_curvature",
    # "gaussian_curvature",
    # "normalized_mean_curvature",
    # "mean_curvature",
    # "hexagon_gaussian_curvature",
    # "curv_rmse",
    # "isotropic_gaussian_curvature",
    # "isotropic_rmse",
    # "isotropic_comparison",
    # "normalized_isotropic_comparison",
    # "areas",
    # "area_dist_error",
    # "area_expansion",
    # "eigenvalues",
    # "eigenvectors",
    # "directors",
    # "angles",
    # "anisotropy",
    # "relative_anisotropy",
    # "out_of_plane_aniosotropy_variation",
    # "z_pc",
    # "z_hex",

    # variables_to_process = [
    #     "gaussian_curvature",
    #     "normalized_gaussian_curvature",
    #     "hexagon_gaussian_curvature",
    #     "mean_curvature",
    #     "normalized_mean_curvature",
    #     "curv_rmse",
    #     "isotropic_gaussian_curvature",
    #     "isotropic_rmse",
    #     "isotropic_comparison",
    #     "normalized_isotropic_comparison",
    #     "area_expansion",
    #     "area_dist_error",
    #     "relative_anisotropy",
    #     "anisotropy",
    #     "out_of_plane_aniosotropy_variation",
    # ]

    files_to_process = [
        "SIM_96_dome",
        "SIM_96_heart",
        "SIM_96_saddle",
        "SIM_96_plane",
        "SIM_600_dome",
        "SIM_600_planar",
        "SIM_600_saddle",
        "EXP_96_saddle",
        "EXP_96_dome",
        "EXP_96_plane",
        "EXP_96_1",
        "EXP_96_2",
        "EXP_96_3",
        "EXP_96_4",
    ]

    variables_to_process = [
        "gaussian_curvature",
        "mean_curvature",
    ]

    ignore_boundaries = {
        "gaussian_curvature": False,
        "normalized_gaussian_curvature": False,
        "hexagon_gaussian_curvature": False,
        "mean_curvature": False,
        "normalized_mean_curvature": False,
        "curv_rmse": False,
        "isotropic_gaussian_curvature": False,
        "normalized_isotropic_gaussian_curvature": False,
        "isotropic_rmse": False,
        "isotropic_comparison": False,
        "normalized_isotropic_comparison": False,
        "area_expansion": False,
        "area_dist_error": False,
        "relative_anisotropy": False,
        "anisotropy": False,
        "out_of_plane_aniosotropy_variation": False,
    }

    interp_folder = "cubic_interp"
    interpolation_type = "cubic"
    processed_data_folder = "/Exp-3-12/"

    interpolate_fx(
        files_to_process,
        variables_to_process,
        ignore_boundaries,
        INTERP_FOLDER=MAIN_INTERP_FOLDER + interp_folder + "/",
        interpolation_type=interpolation_type,
        DATA_FOLDER=MAIN_DATA_FOLDER + processed_data_folder,
    )
    # corresp_independent_variables = [
    #     "point_cloud",
    #     "point_cloud",
    # ]

    # corresp_independent_file = [""]
