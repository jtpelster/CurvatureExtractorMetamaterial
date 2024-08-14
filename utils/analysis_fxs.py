"""
Set of functions about general math analysis, especially the generalized
finite difference derivative method.
"""
from warnings import warn
import math
from time import time
import numpy as np
from utils import point_cloud_utils as pcu


from shapely.geometry import Polygon
from shapely.geometry import Point
from scipy.spatial import ConvexHull

# import scipy.interpolate.Rbf as Interpolator
from scipy.interpolate import RBFInterpolator as Interpolator

from numpy import pi
from numpy import sqrt
from numpy import sin
from numpy import cos
from numpy import log


# %% codecell
def rotation_matrix_to_xy_given_normal(n):
    nx, ny, nz = n[0], n[1], n[2]
    del nz
    nd = np.sqrt(nx ** 2 + ny ** 2)
    Rz = np.array([[nx / nd, ny / nd, 0], [-ny / nd, nx / nd, 0], [0, 0, 1]])

    nprime = Rz * np.array(n)
    npz = nprime[2]
    npx = nprime[0]
    Ry = np.array([[npz, 0, -npx], [0, 1, 0], [npx, 0, npz]])

    error = np.linalg.norm(np.array([[1], [0], [0]]) - Ry * Rz * np.array(n))

    print("error is:" + str(error))


def hexagon_example_points(radius, angle, noise):
    theta = angle * pi / 180
    base_hex = np.array(
        [
            [1, 0],
            [1 / 2, sqrt(3) / 2],
            [-1 / 2, sqrt(3) / 2],
            [-1, 0],
            [-1 / 2, -sqrt(3) / 2],
            [1 / 2, -sqrt(3) / 2],
        ]
    )
    R = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
    final_base_hex = radius * (base_hex @ R)
    return final_base_hex


def gridded_finite_difference(data, h):
    derivative = np.gradient(data, h)
    secx_der = np.gradient(derivative[0], h)
    secy_der = np.gradient(derivative[1], h)

    return [derivative[0], derivative[1], secx_der[0], secx_der[1], secy_der[1]]


def filter_nans_from_row(matrix):
    return matrix[~np.isnan(matrix).any(axis=1)]


def first_order(dx, dy):
    return [dx, dy]


def second_order(dx, dy):
    return [dx, dy, dx ** 2 / 2, dx * dy, dy ** 2 / 2]


def third_order(dx, dy):
    return [
        dx,
        dy,
        dx ** 2 / 2,
        dx * dy,
        dy ** 2 / 2,
        dx ** 3 / 6,
        dx ** 2 * dy / 2,
        dx * dy ** 2 / 2,
        dy ** 3 / 6,
    ]


def fourth_order(dx, dy):
    return [
        dx,
        dy,
        dx ** 2 / 2,
        dx * dy,
        dy ** 2 / 2,
        dx ** 3 / 6,
        dx ** 2 * dy / 2,
        dx * dy ** 2 / 2,
        dy ** 3 / 6,
        dx ** 4 / 12,
        dx ** 3 * dy / 6,
        dx ** 2 * dy ** 2 / 2,
        dx * dy ** 3 / 6,
        dy ** 4 / 12,
    ]


order_dict = {
    1: first_order,
    2: second_order,
    3: third_order,
    4: fourth_order,
}
order_len_dict = {1: 2, 2: 5, 3: 9, 4: 14}
len_order_dict = {y: x for x, y in order_len_dict.items()}


def generalized_finite_difference_calculation(p0, function_table, weights, order):
    """
    Takes in a point p0 and its neighbors with an associated weight and
    estimates the first and second derivatives at p0 in an irregular, arbitrary
    2d grid.
    Based on (Jacquemin et al. 2020 Taylor-Series Expansion Based Numerical Methods)

    Parameters:
        p0 ( list of floats): x0, y0, f(x0,y0) [the evaluated function]
        function_table (list of list of floats): for each p1 to pm, a list of
            the following: [x value, y value, f(x,y)].
            (i.e. an evaluated function of euclidean distance and anisotropy)
    Returns:
        (floats) the first and second derivatives evaluated at p0.
    """

    function_table = np.array(function_table)

    m = len(function_table)
    x0, y0, f0 = p0

    try:
        order_fx = order_dict[order]
    except KeyError:
        raise ValueError(
            "The specified order" + str(order) + "is not currently implemented."
        )

    matrix = np.zeros((m, order_len_dict[order]))
    for data_point in range(m):
        xi, yi, zi = function_table[data_point]
        del zi
        dx = xi - x0
        dy = yi - y0

        matrix[data_point, :] = np.array(order_fx(dx, dy))
    weight_matrix = np.diag(weights)

    # matrix_pseudoinverse = np.linalg.pinv(matrix)
    F = function_table[:, 2] - f0
    comp_det_switch = 0
    if len(function_table[:, 2]) < order_len_dict[order]:
        raise ValueError(
            "Provided points are not enough to estimate the second derivative at this location. Must provide at least"
            + str(order_len_dict[order])
            + " points for order:"
            + str(order)
        )

    if len(function_table[:, 2]) == 5:
        completely_determined_singular_check = np.linalg.det(matrix)
        if completely_determined_singular_check:
            Df = np.linalg.inv(matrix) @ F
            comp_det_switch = 1
        else:
            comp_det_switch = 0

    overdetermined_singular_check = np.linalg.det(matrix.T @ weight_matrix @ matrix)
    if not comp_det_switch:
        if overdetermined_singular_check:
            Df = (
                np.linalg.inv(matrix.T @ weight_matrix @ matrix)
                @ matrix.T
                @ weight_matrix
                @ F
            )
        else:
            warn("Matrix is singular, retrying with pseudoinverse.")
            Df = (
                np.linalg.pinv(matrix.T @ weight_matrix @ matrix)
                @ matrix.T
                @ weight_matrix
                @ F
            )
    # Df = matrix_pseudoinverse @ F
    return Df


def fourth_order_spline(s):
    function = 1 - 6 * s ** 2 + 8 * s ** 3 - 3 * s ** 4
    return function


def constant_weight(x):
    return np.ones((len(x)))


def weight_function(distance, maximum, weight_function_type):
    ##Assumes start is at zero.
    if weight_function_type == "fourth_order":
        weight = fourth_order_spline(distance / maximum)
    if weight_function_type == "constant":
        weight = constant_weight(distance / maximum)
    return weight


def calculate_rmse(derivative, point, neighborhood_points):
    eval_function = order_dict[len_order_dict[len(derivative)]]
    points = neighborhood_points - point

    def rmse_cal(x, y, z):
        z - np.dot(eval_function(x, y), derivative)
        return rmse

    eval = [z - np.dot(eval_function(x, y), derivative) for x, y, z in points]
    rmse = np.sqrt(np.mean(np.array(eval) ** 2))

    return rmse


def generalized_finite_difference(
    point,
    neighborhood_points,
    distances,
    weight_function_type,
    order,
    maxiumum_neighborhood_distance,
):

    # print(distances)
    weight = weight_function(
        distances, maxiumum_neighborhood_distance, weight_function_type
    )
    # print(weight)
    derivative = generalized_finite_difference_calculation(
        point, neighborhood_points, weight, order
    )
    rmse = calculate_rmse(derivative, point, neighborhood_points)
    return derivative, rmse


def generate_convex_hull_polygon(point_cloud):
    convex_hull = ConvexHull(point_cloud[:, 0:2])
    convex_hull_polygon = Polygon((point_cloud[convex_hull.vertices, 0:2]))
    return convex_hull_polygon


def fit_plane_to_edge(edge_points):
    edge_points = np.array(edge_points)
    A = np.hstack((np.ones((len(edge_points), 1)), edge_points[:, :2]))
    y = edge_points[:, 2]

    [a, b, c] = (np.linalg.pinv(A) @ y).T

    def plane(x, y):
        z = a + b * x + c * y
        return z

    return plane


def interpolate_thin_plate(point_cloud):
    fx = Interpolator(point_cloud[:, :2], point_cloud[:, 2])

    def interpolating_fx(x, y):
        return fx(np.array([[x, y]]))

    return interpolating_fx


def select_neighborhood_points(
    neighborhood_points, point, extrapolate_fx="none", convex_hull="none"
):
    if not extrapolate_fx == "none" and convex_hull == "none":
        return ValueError("The convex hull must be provided to generate ghost points.")
    elif extrapolate_fx == "none" and not convex_hull == "none":
        pass

    selected_points = []
    border_point_number = 0

    if not extrapolate_fx == "none":
        for neighborhood_point in neighborhood_points:
            mirrored_pt = 2 * point - neighborhood_point
            if not convex_hull.contains(Point(mirrored_pt[:2])):
                ghost_point = np.append(
                    mirrored_pt[:2], [extrapolate_fx(mirrored_pt[0], mirrored_pt[1])]
                )
                selected_points.append(neighborhood_point)
                selected_points.append(ghost_point)
                border_point_number = border_point_number + 1
            else:
                selected_points.append(neighborhood_point)
            if len(selected_points) == len(neighborhood_points) + 1:
                extra_ghost_point = selected_points.pop()
                del extra_ghost_point
                break
            elif len(selected_points) == len(neighborhood_points):
                break
            elif len(selected_points) > len(neighborhood_points) + 1:
                raise ValueError("There is more than one extra point present.")

    else:
        selected_points = neighborhood_points
        if not convex_hull == "none":
            for neighborhood_point in neighborhood_points:
                mirrored_pt = 2 * point - neighborhood_point
                if not convex_hull.contains(Point(mirrored_pt[:2])):
                    border_point_number = border_point_number + 1

    return selected_points, border_point_number


def derivative_estimator(
    point_cloud,
    neighborhood,
    order=2,
    ghost_point_type="none",
    edge_points=[],
    weight_function="constant",
):

    if neighborhood < ((order + 1) * (order + 2) / 2 - 1):
        raise ValueError(
            "Specified Neighborhood length must match number of unique derivatives."
        )

    if ghost_point_type == "fixed":
        if len(edge_points) == 0:
            ValueError("Fixed boundary condition with no specified edge_points!")
        extrapolate_fx = fit_plane_to_edge(edge_points)

    elif ghost_point_type == "thin_plate":
        extrapolate_fx = interpolate_thin_plate(point_cloud)

    elif ghost_point_type == "none":
        extrapolate_fx = "none"

    else:
        ValueError(
            "Undefined ghost_point_type:"
            + ghost_point_type
            + " Must be fixed, thin_plate, or none"
        )

    derivatives = []
    border_point_numbers = []
    rmse = []
    convex_hull = generate_convex_hull_polygon(point_cloud)
    neighborhood_point_idxs, distances = pcu.n_closest_points(
        point_cloud[:, :2], neighborhood
    )
    maxiumum_neighborhood_distance = np.amax(distances)
    for idx, point in enumerate(point_cloud):
        neighborhood_points, border_point_num = select_neighborhood_points(
            point_cloud[neighborhood_point_idxs[idx]],
            point_cloud[idx],
            extrapolate_fx,
            convex_hull,
        )
        gfd = generalized_finite_difference(
            point,
            neighborhood_points,
            distances[idx],
            weight_function,
            order,
            maxiumum_neighborhood_distance,
        )
        derivatives.append(gfd[0].T)
        rmse.append(gfd[1])
        border_point_numbers.append(border_point_num)
    derivatives = np.array(derivatives)
    border_point_numbers = np.array(border_point_numbers)
    return derivatives, border_point_numbers, rmse
