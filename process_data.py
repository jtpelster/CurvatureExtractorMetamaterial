# %% codecell


# import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from utils import utility as utils
from utils import analysis_fxs as afx
from utils import point_cloud_utils as pcu
import plotly
import re
import plotly.graph_objs as go
import time

###

# %% codecell
# OPTIONS TO CHANGE

# File to load

# Reference file is the file that the hexagons will be extracted from (should be flat)
triangle_ring_reference = {
    0: 0,
    1: 3,
    2: 9,
    3: 12,
    4: 18,
    5: 24,
    6: 30,
    7: 36,
}
triangle_neighborhood_reference = {
    0: 0,
    1: np.sqrt(3) / 3,
    2: 1,
    3: 2 * np.sqrt(3) / 3,
    4: np.sqrt(29 / 12),
    5: np.sqrt(3),
    6: 2,
    7: np.sqrt(13 / 3),
}
triangle_to_hexagon_ring_reference = {
    0: 0,
    3: 1,
    5: 2,
    7: 3,
}
hexagon_neighborhood_reference = {
    0: 0,
    1: 6,
    2: 12,
    3: 18,
    4: 30,
    5: 36,
}
# %% codecell


def evaluate_areas_mesh(point_cloud, hexagon_mesh):
    """
    Calculates the area of a hexagon mesh:

    Param:
        point_cloud (array): x,y,z coordinates of hexagon triangles.
        hexagon_mesh (list of lists): list of hexagons with four triangles with
        three indexes representing the coordinates in point_cloud.
    Returns:
        areas (list of floats):Area of each hexagon: the sum of triangles.
    """
    p = point_cloud
    areas = []
    mesh = []
    for shape in hexagon_mesh:
        shape_area = 0
        for triangle in shape:
            triangle_area = (
                np.linalg.norm(
                    np.cross(
                        p[triangle[1]] - p[triangle[0]], p[triangle[2]] - p[triangle[0]]
                    )
                )
                / 2
            )
            shape_area = triangle_area + shape_area
            mesh += (
                p[triangle[1]].tolist(),
                p[triangle[0]].tolist(),
                p[triangle[2]].tolist(),
            )

        areas.append(shape_area)
    return areas


def plane_fitting_function(M, a, b, c):
    x, y = M
    z = a * x + b * y + c
    return z


# %% codecell
def find_projection_plane(point_cloud, hexagons):
    function_values = []
    for hexagon in hexagons:
        funct_values, covariance = curve_fit(
            plane_fitting_function,
            np.array(point_cloud[hexagon, 0:2]).T,
            np.array(point_cloud[hexagon, 2]).T,
        )
        del covariance
        function_values.append(funct_values)
    return function_values


# %% codecell
def project_points_and_find_perpendicular_point_distance(
    point_cloud, function_values, hexagons
):
    p = point_cloud
    projection_set = []
    perpendicular_distance_set = []
    projected_hexagon_set = []
    for idx_hex, [hexagon, function_value] in enumerate(zip(hexagons, function_values)):
        normal = np.array([-function_value[0], -function_value[1], 1])
        point_on_plane_z = plane_fitting_function(p[0, 0:2], *function_value)
        point_on_plane = np.hstack((p[0, 0:2], [point_on_plane_z]))
        # unit_vector_plane_x = fitting_function()
        projection_point = []
        perpendicular_point_dist = []
        projected_hexagon = []
        casing = []
        points = []
        perpendicular_vector = []
        for idx_pt, point in enumerate(hexagon):
            v = p[point] - point_on_plane
            v_perp = (
                np.dot(v, normal) / (np.linalg.norm(normal) ** 2) * normal
            ).tolist()  # perpendicular to plane
            v_parallel = (
                v - np.dot(v, normal) / (np.linalg.norm(normal) ** 2) * normal
            ).tolist()  # paralell to plane
            projection_point.append((point_on_plane + v_parallel).tolist())
            perpendicular_vector.append(v_perp)
            perpendicular_point_dist.append(np.linalg.norm(v_perp))
            projected_hexagon.append(6 * idx_hex + idx_pt)
            points.append(p[point])

        projection_set.append(projection_point)
        perpendicular_distance_set.append(perpendicular_point_dist)
        projected_hexagon_set.append(projected_hexagon)

    # Perpendicular_distance_set:
    # [Hexagon[Distance]]
    # [[d1,...,d6],....,[d1,...,d6]]
    # projection_set
    # [Hexagon[Point[Coordinates]]]
    # [[[x1,y1,z1],...,[x6,y6,z6]],...,[]]
    # projected_hexagon_set
    # [Hexagon[Point_index_in_projection_set]]
    # [[0,...,5],[6,...],....[...,##]]
    return perpendicular_distance_set, projection_set, projected_hexagon_set


def calculate_hex_projn_areas_perp_dist(point_cloud, hexagon_set):
    function_values = find_projection_plane(point_cloud, hexagon_set)
    # rotate plane to generate the projection_point_set on the xy plane
    (
        perpendicular_distance_set,
        projection_set,
        projected_hexagon_set,
    ) = project_points_and_find_perpendicular_point_distance(
        point_cloud, function_values, hexagon_set
    )

    projection_point_set = []
    for points in projection_set:
        projection_point_set += points
    projection_point_set = np.array(projection_point_set)

    projection_hexagon_mesh = pcu.generate_hex_mesh(
        projection_point_set, projected_hexagon_set
    )
    initial_projected_hexagon_areas = evaluate_areas_mesh(
        projection_point_set, projection_hexagon_mesh
    )
    perpendicular_dist_rmse = [
        np.sqrt(np.mean(np.array(distances)) ** 2)
        for distances in perpendicular_distance_set
    ]
    return (
        initial_projected_hexagon_areas,
        perpendicular_dist_rmse,
    )  # projection_point_set, projection_hexagon_mesh


# %% codecell
def isotropic_gaussian(coordinates, hexagon_relative_area, order, neighborhood):
    # DOUBLE CHECK:
    # TAKE DERIVATIVE WRT ORIGINAL COORDINATES?
    insides = np.log(hexagon_relative_area)
    inside = np.reshape(insides, (-1, 1))
    log_area_point_cloud = np.hstack((coordinates[:, :2], inside))
    derivative, border_point_num, rmse = afx.derivative_estimator(
        log_area_point_cloud, neighborhood, order=order, weight_function="constant",
    )

    Ax, Ay, Axx, Axy, Ayy = derivative[:, :5].T
    del Ax, Ay, Axy
    laplacian = Axx + Ayy
    curvature = -laplacian / hexagon_relative_area
    return curvature, rmse


# %% codecell
def prin_comp_analysis(points):
    points = np.array(points).T
    mean = np.mean(points, axis=1)
    centered_points = (points.T - mean.T).T
    covariance = np.cov(centered_points)

    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    if not (len(eigenvalues) == 2 or len(eigenvalues) == 3):
        raise ValueError("This function doesn't handle more than three dimesions.")
    return eigenvalues, eigenvectors



# %% codecell
def calculate_anisotropy(hexagon_set, point_cloud):
    eigenvalues_list = []
    eigenvector_list = []
    directors_list = []
    angles_list = []
    anisotropies_list = []
    special_vector_list = []
    for hexagon_idxs in hexagon_set:
        hexagon = point_cloud[hexagon_idxs]
        data = np.array(hexagon)
        eigenvalues, eigenvectors = prin_comp_analysis(data)
        if len(eigenvalues) == 2:
            lambda1, lambda2 = eigenvalues
        elif len(eigenvalues) == 3:
            lambda1, lambda2, lambda3 = eigenvalues
        else:
            raise ValueError("This program doesn't handle more than three dimesions.")
        stretch_major = np.sqrt(lambda1)
        stretch_minor = np.sqrt(lambda2)
        anisotropy = stretch_major / stretch_minor
        relative_anisotropy = (
            (stretch_major - stretch_minor) * 2 / (stretch_major + stretch_minor)
        )
        major = [[0, 0, 0], eigenvectors[:, 0].tolist()]
        minor = [[0, 0, 0], eigenvectors[:, 1].tolist()]
        out_of_plane = [[0, 0, 0], eigenvectors[:, 2].tolist()]
        major_phi = np.arctan2(eigenvectors[0, 0], eigenvectors[1, 0])
        minor_phi = np.arctan2(eigenvectors[0, 1], eigenvectors[1, 1])
        eigenvalues_list.append([lambda1, lambda2, lambda3])
        eigenvector_list.append(
            [
                eigenvectors[:, 0].tolist(),
                eigenvectors[:, 1].tolist(),
                eigenvectors[:, 2].tolist(),
            ]
        )
        theta = major_phi
        print(major)
        print(minor)
        #req = np.sqrt(major*minor)/np.pi
        req = 2 #THIS IS A BAD FIX
        directors_list.append([major, minor, out_of_plane])
        angles_list.append([major_phi, minor_phi])
        anisotropies_list.append([anisotropy, relative_anisotropy])
        special_vector_list.append([anisotropy*np.cos(theta),anisotropy*np.sin(theta),np.sqrt(req**2-anisotropy**2)])
    return (
        eigenvalues_list,
        eigenvector_list,
        directors_list,
        angles_list,
        anisotropies_list,
        special_vector_list
    )


# %% codecell


def sort_selected_files(all_files):
    """
    Sorts given files by ref or comparison using json value "data_type"
    """
    reference_files = []
    comparison_files = []
    for file in all_files:
        data_type = utils.get_variables_from_json(file, "data_type")
        if data_type == "reference":
            reference_files.append(file)
        elif data_type == "comparison":
            comparison_files.append(file)
        else:
            raise ValueError(
                "Read file: "
                + file
                + " does not have a declared comparison or reference type."
            )

    return reference_files, comparison_files


def load_ref_data(file):
    (
        point_cloud,
        hexagons,
        hex_coordinates,
        edge_point_idx,
    ) = utils.get_variables_from_json(
        file, ["point_cloud", "hexagons", "hex_coordinates", "edge_point_idx"]
    )
    point_cloud = np.array(point_cloud)
    edge_coords = point_cloud[edge_point_idx]
    return point_cloud, hexagons, hex_coordinates, edge_point_idx, edge_coords


def gaussian_curvature_fx(dx, dy, dxx, dxy, dyy):
    K = (dxx * dyy - (dxy ** 2)) / (1 + (dx ** 2) + (dy ** 2)) ** 2
    return K


def mean_curvature_fx(dx, dy, dxx, dxy, dyy):
    H_temp = (dx ** 2 + 1) * dyy - 2 * dx * dy * dxy + (dy ** 2 + 1) * dxx
    H = -H_temp / (2 * (dx ** 2 + dy ** 2 + 1) ** (1.5))
    return H


def normalize_gaussian(curvature, length_scale):
    normalized_gaussian = [
        k / np.abs(k) * np.sqrt(np.abs(k)) * length_scale if np.abs(k) > 0 else 0
        for k in curvature
    ]
    return normalized_gaussian


def calculate_gaussian_and_mean_curvature(
    point_cloud, order, neighborhood, is_fixed, edge_points, length_scale=1
):

    if is_fixed:
        ghost_point_type = "fixed"
    else:
        ghost_point_type = "thin_plate"

    derivative, border_point_num, rmse = afx.derivative_estimator(
        point_cloud,
        neighborhood,
        order,
        ghost_point_type,
        edge_points,
        weight_function="constant",
    )
    dx, dy, dxx, dxy, dyy = derivative[:, :5].T
    gaussian_curvature = gaussian_curvature_fx(dx, dy, dxx, dxy, dyy)
    mean_curvature = mean_curvature_fx(dx, dy, dxx, dxy, dyy)

    normalized_gaussian_curvature = normalize_gaussian(gaussian_curvature, length_scale)
    normalized_mean_curvature = mean_curvature * length_scale
    return (
        gaussian_curvature,
        normalized_gaussian_curvature,
        mean_curvature,
        normalized_mean_curvature,
        rmse,
    )


def average_over_hexagons(triangle_value, hexagons):
    hexagon_value = [np.mean(triangle_value[hexagon]) for hexagon in hexagons]
    return hexagon_value


def load_comparison_data(file):
    (
        point_cloud,
        hexagons,
        hex_coordinates,
        is_fixed,
        edge_points,
        length_scale,
    ) = utils.get_variables_from_json(
        file,
        [
            "point_cloud",
            "hexagons",
            "hex_coordinates",
            "fixed",
            "edge_points",
            "length_scale",
        ],
    )
    # FROM REFERENCE FILE LOAD INITIAL AREAS AND ORIGINAL HEX POINT CLOUD
    return point_cloud, hexagons, hex_coordinates, is_fixed, edge_points, length_scale


def get_reference_file(comparison_file):
    try:
        reference = utils.get_variables_from_json(comparison_file, "reference_file")
    except KeyError as ke:
        raise KeyError("reference_filename is undefined in: " + comparison_file) from ke
    return SAVING_PATH + reference


def process_reference_files(
    reference_file, derivative_order, derivative_neighborhood,
):
    ref_variable_names_to_save = [
        "gaussian_curvature",
        "mean_curvature",
        "curv_rmse",
        "areas",
        "area_dist_error",
    ]

    (
        point_cloud,
        hexagons,
        hex_coordinates,
        edge_point_idx,
        edge_points,
    ) = load_ref_data(reference_file)

    point_cloud = np.array(point_cloud)

    (
        gaussian_curvature,
        normalized_gaussian_curvature,
        mean_curvature,
        normalized_mean_curvature,
        curv_rmse,
    ) = calculate_gaussian_and_mean_curvature(
        point_cloud, derivative_order, derivative_neighborhood, True, edge_points
    )
    hexagon_gaussian_curvature = average_over_hexagons(gaussian_curvature, hexagons)
    areas, distances = calculate_hex_projn_areas_perp_dist(point_cloud, hexagons)

    (eigenvalues, eigenvectors, directors, angles, anisotropies, svl) = calculate_anisotropy(
        hexagons, point_cloud
    )
    eigenvalues = np.array(eigenvalues)
    out_of_plane_aniosotropy_variation = eigenvalues[:, 2] / np.amin(
        eigenvalues[:, :2], 1
    )
    anisotropies = np.array(anisotropies)
    hex_coordinates = np.array(hex_coordinates)
    saving_dict = {
        "normalized_gaussian_curvature": normalized_gaussian_curvature,
        "gaussian_curvature": gaussian_curvature,
        "normalized_mean_curvature": normalized_mean_curvature,
        "mean_curvature": mean_curvature,
        "hexagon_gaussian_curvature": hexagon_gaussian_curvature,
        "curv_rmse": curv_rmse,
        "areas": areas,
        "area_dist_error": distances,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "directors": directors,
        "angles": angles,
        "anisotropy": anisotropies[:, 0],
        "relative_anisotropy": anisotropies[:, 1],
        "out_of_plane_aniosotropy_variation": out_of_plane_aniosotropy_variation,
        "z_pc": point_cloud[:, 2],
        "z_hex": hex_coordinates[:, 2],
    }
    ref_variables_to_save = [saving_dict[name] for name in ref_variable_names_to_save]
    utils.save_json(
        reference_file, ref_variable_names_to_save, ref_variables_to_save, 0,
    )


def process_comparison_files(
    comparison_file,
    derivative_order,
    derivative_neighborhood,
    isotropic_derivative_order,
    isotropic_neighborhood,
):

    variable_names_to_save = [
        "normalized_gaussian_curvature",
        "gaussian_curvature",
        "normalized_mean_curvature",
        "mean_curvature",
        "hexagon_gaussian_curvature",
        "curv_rmse",
        "isotropic_gaussian_curvature",
        "normalized_isotropic_gaussian_curvature",
        "isotropic_rmse",
        "isotropic_comparison",
        "normalized_isotropic_comparison",
        "areas",
        "area_dist_error",
        "area_expansion",
        "eigenvalues",
        "eigenvectors",
        "directors",
        "angles",
        "anisotropy",
        "relative_anisotropy",
        "out_of_plane_aniosotropy_variation",
        "z_pc",
        "z_hex",
    ]

    (
        point_cloud,
        hexagons,
        hex_coordinates,
        is_fixed,
        edge_points,
        length_scale,
    ) = load_comparison_data(comparison_file)
    del hex_coordinates  # Unneeded for now

    point_cloud = np.array(point_cloud)
    (
        gaussian_curvature,
        normalized_gaussian_curvature,
        mean_curvature,
        normalized_mean_curvature,
        curv_rmse,
    ) = calculate_gaussian_and_mean_curvature(
        point_cloud,
        derivative_order,
        derivative_neighborhood,
        is_fixed,
        edge_points,
        length_scale,
    )
    ref_file = get_reference_file(comparison_file)
    initial_areas, initial_hex_coordinates, hexagons = utils.get_variables_from_json(
        ref_file, ["areas", "hex_coordinates", "hexagons"]
    )
    hexagon_gaussian_curvature = average_over_hexagons(gaussian_curvature, hexagons)
    normalized_hexagon_gaussian_curvature = normalize_gaussian(
        hexagon_gaussian_curvature, length_scale
    )
    initial_hex_coordinates = np.array(initial_hex_coordinates)
    current_areas, distances = calculate_hex_projn_areas_perp_dist(
        point_cloud, hexagons
    )
    area_expansion = np.array(current_areas) / np.array(initial_areas)
    isotropic_gaussian_curvature, iso_rmse = isotropic_gaussian(
        initial_hex_coordinates,
        area_expansion,
        isotropic_derivative_order,
        isotropic_neighborhood,
    )
    normalized_isotropic_gaussian_curvature = normalize_gaussian(
        isotropic_gaussian_curvature, length_scale
    )
    isotropic_comparison_max = np.amax(normalized_hexagon_gaussian_curvature)
    isotropic_comparison = np.array(hexagon_gaussian_curvature) - np.array(
        isotropic_gaussian_curvature
    )
    isotropic_comparison = normalize_gaussian(isotropic_comparison, length_scale)
    normalized_isotropic_comparison = isotropic_comparison / isotropic_comparison_max

    (eigenvalues, eigenvectors, directors, angles, anisotropies,svl) = calculate_anisotropy(
        hexagons, point_cloud
    )
    eigenvalues = np.array(eigenvalues)
    out_of_plane_aniosotropy_variation = eigenvalues[:, 2] / np.amin(
        eigenvalues[:, :2], 1
    )

    anisotropies = np.array(anisotropies)

    saving_dict = {
        "normalized_gaussian_curvature": normalized_gaussian_curvature,
        "gaussian_curvature": gaussian_curvature,
        "normalized_mean_curvature": normalized_mean_curvature,
        "mean_curvature": mean_curvature,
        "hexagon_gaussian_curvature": hexagon_gaussian_curvature,
        "curv_rmse": curv_rmse,
        "isotropic_gaussian_curvature": isotropic_gaussian_curvature,
        "normalized_isotropic_gaussian_curvature": normalized_isotropic_gaussian_curvature,
        "isotropic_rmse": iso_rmse,
        "isotropic_comparison": isotropic_comparison,
        "normalized_isotropic_comparison": normalized_isotropic_comparison,
        "areas": current_areas,
        "area_dist_error": distances,
        "area_expansion": area_expansion,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "directors": directors,
        "angles": angles,
        "anisotropy": anisotropies[:, 0],
        "relative_anisotropy": anisotropies[:, 1],
        "out_of_plane_aniosotropy_variation": out_of_plane_aniosotropy_variation,
        "z_pc": point_cloud[:, 2],
        "z_hex": initial_hex_coordinates[:, 2],
        "special_vector": svl
    }

    variables_to_save = [saving_dict[name] for name in variable_names_to_save]
    utils.save_json(
        comparison_file, variable_names_to_save, variables_to_save, 0,
    )


def process(
    foldername,
    files_to_analyze,
    derivative_order=3,
    derivative_neighborhood=12,
    isotropic_derivative_order=3,
    isotropic_neighborhood=12,
):

    global SAVING_PATH

    data_path = "./formatted_data/"
    process_path = "./processed_data/"
    SAVING_PATH = process_path + foldername + "/"

    for file in files_to_analyze:
        utils.copy_json(data_path + file, SAVING_PATH + file)

    absolute_files = []
    for file in files_to_analyze:
        absolute_files.append(SAVING_PATH + file)

    reference_files, comparison_files = sort_selected_files(absolute_files)
    for reference_file in reference_files:
        process_reference_files(
            reference_file, derivative_order, derivative_neighborhood,
        )
    for comparison_file in comparison_files:
        process_comparison_files(
            comparison_file,
            derivative_order,
            derivative_neighborhood,
            isotropic_derivative_order,
            isotropic_neighborhood,
        )
    print("Processing Complete")


if __name__ == "__main__":
    files_to_analyze = [
        "SIM_96_REF",
        "SIM_600_REF",
        "EXP_96_REF",
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
    derivative_order=3
    derivative_neighborhood=12
    # files_to_analyze = [
    #     "EXP_96_REF",
    #     "EXP_96_1",
    #     "EXP_96_2",
    #     "EXP_96_3",
    #     "EXP_96_4",
    # ]

    filename = "Exp-3-12"

    process(
        filename, files_to_analyze, derivative_order, derivative_neighborhood
    )
