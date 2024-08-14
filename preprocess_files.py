"""
    This script preprocesses simulation and experimental data into the same
    format.
    The user specifies the filename and reference file.
    The hexagons are calculated for each of the files.
"""
import numpy as np
import open3d as otd

from warnings import warn

from numpy import cos
from numpy import sin

from utils import utility as utils
from utils import point_cloud_utils as pcu

# importlib.reload()

# Ask user for parameters


# %% codecell
# First step: load and format data
def load_data(dataname, UNITY_FRAME_ROTATION_ANGLE, is_unity):
    """
    Loads data in the format x, y, z,
    First line must not include information
    Converts from unity units to microns if specified,
    Rotates unity simulation to match experimental frame
    Moves frame such that minimum x value and y value are 0

        Parameters:
            dataname (string): file name in current folder containing dataname
            is_unity (bool): whether data is in unity frame and units

        Returns:
            data (np array): Array of coordinates with columns x,y,z and
            rows by point index.
    """
    # Takes in a filename and returns the data in an array
    data = np.loadtxt(dataname, skiprows=1, delimiter=",")

    # Scale and Rotate Unity Frame to Experimental Frame
    if is_unity:
        # Match units
        data = data * UNITY_UNITS_TO_MICRONS

        # Rotate
        theta = UNITY_FRAME_ROTATION_ANGLE
        theta = theta * np.pi / 180
        R = np.array([[cos(theta), sin(theta)], [-sin(theta), cos(theta)]])
        data[:, 0:2] = data[:, 0:2] @ R

    # Translate, such that the minimum x and y values are zero
    data_xmin = min(data[:, 0])
    data_ymin = min(data[:, 1])

    data[:, 0] = data[:, 0] - data_xmin
    data[:, 1] = data[:, 1] - data_ymin
    return data


# %% codecell
# Calculate hexagons for each file
def compute_radius(cd, use_default_radius, radius_scale, manual_radius):
    """
    Returns the radius used for determining whether a point is part of a given
        hexagon. Either the mean of nearest distances times a user given factor.
        Or a user supplied radius.
    parameters:
        cd (open3d point cloud)
        use_default_radius (bool): Use calculated radius or user supplied radius.
        radius_scale (float): Value to scale calculated radius by.
        manual_radius (float): Manually supplied radius.
    Returns:
        rad (float): Calculated radius for finding hexagons.
    """
    nearest_dist = cd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(nearest_dist)
    default_radius = radius_scale * avg_dist

    if use_default_radius:
        rad = default_radius
    else:
        rad = manual_radius
    return rad


def compute_max_nearest_dist(cd):
    """
    Finds the closest distances to each point,
    Takes the maximum over all points.

    Parameters:
        cd (open3d point cloud): doesn't require normals
        radius_scale (float): Value to scale radius by to capture all points in
            closest three points of hexagon

    Returns:
        rad (float): radius used as cutoff for inner three points of hexagon set
        max_nearest_dist (float): maximum of nearest distances of all points in cd
        avg_dist (float): average of the nearest distances of all points in cd
    """
    # Finds the closest distance for each point. Takes the maximum.
    nearest_dist = cd.compute_nearest_neighbor_distance()
    # Convert doublevector into array to find maximum nearest distance
    dist_array = np.zeros(np.size(nearest_dist))
    for idx, d in enumerate(nearest_dist):
        dist_array[idx] = d
    max_nearest_dist = dist_array.max()
    return max_nearest_dist


# %% codecell


# %% codecell
# Find the angles between each of the points (should be symetric matrix)
def angle_array(point_cloud):
    """
    Calculates the positive angle between two different points, given two indeces.
    Parameters:
        point_cloud (array): 3d positions of points x,y,z are columns, point
        indexes are rows.

    Returns:
        angle_ref (array): compares row index point to column index point,
        giving the positive angle between them.
    """
    angle_ref = np.zeros([len(point_cloud), len(point_cloud)])
    for center_idx, center_point in enumerate(point_cloud):
        for other_idx, other_point in enumerate(point_cloud):

            vector = other_point - center_point
            neg_pi_to_pi_angle = np.arctan2(vector[1], vector[0])

            if neg_pi_to_pi_angle < 0:
                zero_to_two_pi_angle = 2 * np.pi + neg_pi_to_pi_angle
            else:
                zero_to_two_pi_angle = neg_pi_to_pi_angle
            angle_ref[center_idx, other_idx] = zero_to_two_pi_angle

    # angle_ref: The row index is the first point,
    # the column index is the comparision point.
    # Value is the angle between the points. Should be symetric.
    return angle_ref


# %% codecell
def sort_by_angle_remove_points_outside_of_hexagon_and_number(
    angle_ref, point_set, point_cloud, maxdist, dist, HEXAGON_FUDGE_FACTOR
):
    """
    Docstring
    """
    # Sort each of the given points by angle.
    # Filters out any points outside the size of an expected hexagon.
    # This allows for finding individual hexagons.

    # Declare initial arrays
    number_of_points = len(point_set[0])
    points_angle = np.zeros([len(point_cloud), number_of_points])
    sorted_angle_idx = np.zeros([len(point_cloud), number_of_points])
    closest_points_sorted_by_angle = np.zeros([len(point_cloud), number_of_points])

    cleaned_indexes_of_closest_points = []
    for center_pt, points in enumerate(point_set):
        for pt_num, point in enumerate(points):

            # If point distance is outside that of a hexagon
            if (
                dist[center_pt, point]
                > maxdist
                + 2 * maxdist * np.sin(30 * np.pi / 180)
                + maxdist * HEXAGON_FUDGE_FACTOR
            ):
                points_angle[center_pt, pt_num] = float("NaN")
            else:
                points_angle[center_pt, pt_num] = angle_ref[center_pt, point]

            sorted_angle_idx[center_pt, 0:number_of_points] = np.argsort(
                points_angle[center_pt, :]
            )

            sorted_angle_temp = np.zeros(number_of_points)

            for count, idx in enumerate(sorted_angle_idx[center_pt].astype(int)):
                if np.isnan(points_angle[center_pt, idx]):
                    sorted_angle_temp[count] = float("NaN")
                else:
                    sorted_angle_temp[count] = points[idx]

        # Sorted angle contains 12 closest points

        closest_points_sorted_by_angle[center_pt, :number_of_points] = sorted_angle_temp
        cleaned_indexes_of_closest_points.append(
            closest_points_sorted_by_angle[
                center_pt,
                np.where(~np.isnan(closest_points_sorted_by_angle[center_pt])),
            ]
            .astype(int)
            .tolist()[0]
        )

    return cleaned_indexes_of_closest_points


# %% codecell
def angle_difference(angle1, angle2):
    """
    subtracts angle1 from angle2. Returns the positive angle difference.

    parameters:
        angle1 (float between 0 and 2 pi): the angle being subtracted From
        angle2 (float between 0 and 2 pi): the angle subtracting

    returns:
        difference (float)
    """
    if angle2 > angle1:
        difference = angle1 + 2 * np.pi - angle2
    else:
        difference = angle1 - angle2
    return difference


# %% codecell
def is_between_angles_less_than_pi_apart_counter_clockwise(
    lower_angle, upper_angle, comparison_angle
):
    """
    Checks if comparision angle is between two angles that are less than pi apart
    in the CCW direction.
    Parameters:
        lower_angle (float between 0 and 2 pi): The lower of the two angles
        upper_angle (float between 0 and 2 pi): The upper of the two angles
        comparison_angle (float between 0 and 2 pi): angle being checked if
            between the given angles

    Returns:
        is_between (bool): Whether the angle is between given angles.
    """
    # Checks if comparision angle is between two angles that are less than
    # pi apart in the CCW direction.

    is_between = 0
    if -upper_angle + lower_angle > np.pi:
        if comparison_angle > lower_angle or comparison_angle < upper_angle:
            is_between = 1
    else:
        if upper_angle < comparison_angle < lower_angle:
            is_between = 1
    return is_between


def generate_full_hexagon_set(
    rad, closest_points_sorted_by_angle_array, angle_ref, dist
):

    """
    Takes in a set of closest points for each point up to 12 long, sorted by angle
    and finds all the hexagons that are connected to that point.
    Parameters:
        rad (float): radius indicating expected distance between adjacent
            points in a hexagon
        radius_scale (float): How many times over the expected radius to accept
            points as adjacent to current point.
        closest_points_sorted_by_angle_array (list or array):

        angle_ref (array): positive angle between two points given by
            indices
        dist (array): distance between two points given by indices

    """

    full_hexagon_set = []
    hexagon_ref = []

    for center_pt, indexes in enumerate(closest_points_sorted_by_angle_array):
        del indexes
        closest_points_sorted_by_angle = np.array(closest_points_sorted_by_angle_array[
            center_pt
        ])
        closest_points_sorted_by_angle = np.array(closest_points_sorted_by_angle)

        ## Find the three closest points that are within a certain distance

        # Closest points within adjacency radius
        closest_within = closest_points_sorted_by_angle[
            np.where(dist[center_pt, closest_points_sorted_by_angle.astype(int)] < rad)[
                0
            ]
        ].astype("int")
        # Three closest points
        closest_three_points = closest_within[
            dist[center_pt, closest_within].argsort()[:3]
        ]
        # Sort points by angle
        closest_three_points = closest_three_points[
            angle_ref[center_pt, closest_three_points].argsort()
        ]

        for idx, point in enumerate(closest_three_points):

            # Identify next index, next index after the end is 0
            if idx + 1 > len(closest_three_points) - 1:
                next_idx = 0
            else:
                next_idx = idx + 1

            next_point = closest_three_points[next_idx]

            current_angle = angle_ref[center_pt, point]
            next_angle = angle_ref[center_pt, next_point]

            # Hexagon is composed of the center point,
            # two closest points and three points between the closest points
            hexagon = []

            hexagon.append(center_pt)
            hexagon.append(point)
            if angle_difference(next_angle, current_angle) < np.pi:
                hexagon.append(next_point)

                unfiltered_points_between = []
                for near_point in closest_points_sorted_by_angle:
                    if is_between_angles_less_than_pi_apart_counter_clockwise(
                        current_angle, next_angle, angle_ref[center_pt, near_point]
                    ):
                        unfiltered_points_between.append(near_point)

                points_between = []
                for point_between in unfiltered_points_between:
                    if point_between in hexagon:
                        pass
                    else:
                        points_between.append(point_between)
                points_between = np.array(points_between)

                if len(points_between) != 0:
                    closest_three_points_between_hexagon_boundary = points_between[
                        dist[center_pt, points_between].argsort()[:3]
                    ]
                    hexagon = (
                        hexagon + closest_three_points_between_hexagon_boundary.tolist()
                    )

            switch = True
            if len(hexagon) == 6:
                hexagon_ref.append(hexagon)
                for full_hexagon in full_hexagon_set:
                    if set(hexagon) == set(full_hexagon):
                        switch = False

                if switch:
                    full_hexagon_set.append(hexagon)
    return full_hexagon_set


def get_centers_of_each_hexagon(point_cloud, full_hexagon_set):
    found_vertexes = []
    for hexagon in full_hexagon_set:
        found_vertexes.append((sum(point_cloud[hexagon]) / len(hexagon)).tolist())
    return found_vertexes


def filter_hexagon_centers(radius, full_hexagon_set, hexagon_centers):

    filter_radius = radius / 2

    centers = []
    filtered_full_hexagons = []
    for center, hexagon in zip(hexagon_centers, full_hexagon_set):
        add_switch = 1
        for unfiltered_center in hexagon_centers:
            if (
                np.linalg.norm(np.array(center) - np.array(unfiltered_center))
                - filter_radius
                < 0
            ) and center != unfiltered_center:
                add_switch = 0
        if add_switch == 1:
            centers.append(center)
            filtered_full_hexagons.append(hexagon)
    return filtered_full_hexagons


def generate_open3d_point_cloud(point_cloud):
    """
        Converts points in array to open3d point_cloud with normals.

        Parameters:
            point_cloud (np array): array of points with columns x,y,z and
                rows as point index.

        Returns:
            cd (Open3d point cloud): Point cloud with calculated normals
        """
    # Takes in a list of coordinates and converts to an open3d point_cloud
    cd = otd.geometry.PointCloud()
    cd.points = otd.utility.Vector3dVector(point_cloud)
    cd.estimate_normals()
    cd.orient_normals_consistent_tangent_plane(k=1)
    return cd


def calculate_normals(point_cloud):
    """
    Calculates the normals of a point cloud. This is done using open3d which does
    so by calculating the eigenvectors of the covariance matrix for some neighborhood.
    """
    cd = otd.geometry.PointCloud()
    cd.points = otd.utility.Vector3dVector(point_cloud)
    cd.estimate_normals()
    cd.orient_normals_consistent_tangent_plane(k=1)
    normals = np.asarray(cd.normals[:])
    return normals


def identify_hexagons(
    point_cloud, use_default_radius, radius_scale, manual_radius, HEXAGON_FUDGE_FACTOR
):
    cd = generate_open3d_point_cloud(point_cloud)
    rad = compute_radius(cd, use_default_radius, radius_scale, manual_radius)
    maxdist = compute_max_nearest_dist(cd)
    dist = pcu.dist_array(point_cloud)
    angle_ref = angle_array(point_cloud)

    twelve_closest_points, distances = pcu.n_closest_points(point_cloud, 12)
    del distances
    closest_points_sorted_by_angle = sort_by_angle_remove_points_outside_of_hexagon_and_number(
        angle_ref,
        twelve_closest_points,
        point_cloud,
        maxdist,
        dist,
        HEXAGON_FUDGE_FACTOR,
    )
    full_hexagon_set = generate_full_hexagon_set(
        rad, closest_points_sorted_by_angle, angle_ref, dist,
    )
    unfiltered_centers = get_centers_of_each_hexagon(point_cloud, full_hexagon_set)
    filtered_hexagon_set = filter_hexagon_centers(
        rad, full_hexagon_set, unfiltered_centers
    )

    return filtered_hexagon_set


def extract_edges(coordinates, hexagon_set, hexagon_edge_number_classification):
    hexagon_point_list = np.concatenate(hexagon_set).tolist()
    edge_point_coordinates = []
    edge_point_indexes = []

    for idx, center_point in enumerate(coordinates):
        if hexagon_point_list.count(idx) <= hexagon_edge_number_classification:
            edge_point_coordinates.append(coordinates[idx])
            edge_point_indexes.append(idx)
    edge_point_coordinates = np.array(edge_point_coordinates)
    edge_point_indexes = np.array(edge_point_indexes)
    inner_points = [i for i in range(len(coordinates)) if i not in edge_point_indexes]
    return edge_point_coordinates, edge_point_indexes, inner_points


def extract_edge_hexagons(point_cloud, hexagons, hexagon_edge_class_number):
    # Takes in a point cloud and hexagon list of idxs and outputs the index of the hexagons on
    # the edges
    coords_1, class_hexagon_edge_points, inner_points = extract_edges(
        point_cloud, hexagons, hexagon_edge_class_number
    )
    del coords_1, inner_points

    outer_hexagons = []
    for idx, hexagon in enumerate(hexagons):
        if any(x in hexagon for x in class_hexagon_edge_points):
            outer_hexagons.append(idx)
    inner_hexagons = [i for i in range(len(hexagons)) if i not in outer_hexagons]
    return outer_hexagons, inner_hexagons


# %% codecell


def preprocess(
    reference_files,
    is_reference_unity,
    saved_reference_names,
    comparison_files,
    saved_comparison_file_names,
    associated_reference_filenames,
    is_comparison_unity,
    is_comparison_fixed,
    use_default_radius,
    radius_scale,
    manual_radius,
    hexagon_edge_number_classification,
    UNITY_FRAME_ROTATION_ANGLE,
    HEXAGON_FUDGE_FACTOR,
):
    hexagon_edge_class_number = hexagon_edge_number_classification
    global UNITY_UNITS_TO_MICRONS

    data_path = "./formatted_data/"
    original_file_path = "./original_data/"
    UNITY_UNITS_TO_MICRONS = 100

    for reference_file, saving_name, is_unity in zip(
        reference_files, saved_reference_names, is_reference_unity
    ):

        point_cloud = load_data(
            original_file_path + reference_file, UNITY_FRAME_ROTATION_ANGLE, is_unity
        )
        length_scale = np.amax(point_cloud[:, :2])
        normals = calculate_normals(point_cloud)
        hexagons = identify_hexagons(
            point_cloud,
            use_default_radius,
            radius_scale,
            manual_radius,
            HEXAGON_FUDGE_FACTOR,
        )

        hex_mesh = pcu.generate_hex_mesh(point_cloud, hexagons)
        hex_coordinates = get_centers_of_each_hexagon(point_cloud, hexagons)

        edge_point_coords, edge_point_idx, inner_points = extract_edges(
            point_cloud, hexagons, hexagon_edge_number_classification
        )

        outer_hexagons, inner_hexagons = extract_edge_hexagons(
            point_cloud, hexagons, hexagon_edge_class_number
        )

        utils.save_json(
            data_path + saving_name,
            [
                "data_type",
                "reference_file",
                "length_scale",
                "point_cloud",
                "normals",
                "hexagons",
                "hex_coordinates",
                "hex_mesh",
                "edge_points",
                "edge_point_idx",
                "inner_points",
                "outer_hexagons",
                "inner_hexagons",
            ],
            [
                "reference",
                "none",
                length_scale,
                point_cloud,
                normals,
                hexagons,
                hex_coordinates,
                hex_mesh,
                edge_point_coords,
                edge_point_idx,
                inner_points,
                outer_hexagons,
                inner_hexagons,
            ],
            1,
        )

    for comparison_file, saving_name, reference_filename, is_unity, is_fixed in zip(
        comparison_files,
        saved_comparison_file_names,
        associated_reference_filenames,
        is_comparison_unity,
        is_comparison_fixed,
    ):

        point_cloud = load_data(
            original_file_path + comparison_file, UNITY_FRAME_ROTATION_ANGLE, is_unity
        )

        (
            length_scale,
            hexagons,
            hex_mesh,
            edge_point_coords,
            edge_point_idx,
            inner_points,
            outer_hexagons,
            inner_hexagons,
        ) = utils.get_variables_from_json(
            data_path + reference_filename,
            [
                "length_scale",
                "hexagons",
                "hex_mesh",
                "edge_points",
                "edge_point_idx",
                "inner_points",
                "outer_hexagons",
                "inner_hexagons",
            ],
        )
        try:
            hex_coordinates = get_centers_of_each_hexagon(point_cloud, hexagons)
        except IndexError:
            hex_coordinates = "none"
            warn(
                "Hex_coordinates could not be created due to reference point mismatch."
            )

        utils.save_json(
            data_path + saving_name,
            [
                "data_type",
                "reference_file",
                "length_scale",
                "fixed",
                "point_cloud",
                "normals",
                "hexagons",
                "hex_coordinates",
                "hex_mesh",
                "edge_points",
                "edge_point_idx",
                "inner_points",
                "outer_hexagons",
                "inner_hexagons",
            ],
            [
                "comparison",
                reference_filename,
                length_scale,
                is_fixed,
                point_cloud,
                normals,
                hexagons,
                hex_coordinates,
                hex_mesh,
                edge_point_coords,
                edge_point_idx,
                inner_points,
                outer_hexagons,
                inner_hexagons,
            ],
            1,
        )
        print("Preprocessing Complete")


if __name__ == "__main__":
    reference_files = [
        "COM_96Panels_NoActuation.txt",
        "COM_600Panels_NoActuation.txt",
        "Point cloud of metamaterial sheet1_118.csv",
    ]
    is_reference_unity = [True, True, False]
    saved_reference_names = ["SIM_96_REF", "SIM_600_REF", "EXP_96_REF"]
    # Place the file in the same folder as the program
    # This should include the reference file
    comparison_files = [
        "COM_dome.txt",
        "COM_heart_on_snowflake.txt",
        "COM_monkey_saddle.txt",
        "COM_planar.txt",
        "dome_600Panels.txt",
        "planar_600Panels.txt",
        "saddle_600Panels.txt",
        "Point cloud of metamaterial sheet_Image 6_saddle2_fixed.csv",
        "Point cloud of metamaterial sheet_Image 57_sphere2_3best.csv",
        "Point cloud of metamaterial sheet_Image 52_2D_Temp.csv",
        "Point cloud of metamaterial sheet1_118.csv",
        "Point cloud of metamaterial sheet2_105.csv",
        "Point cloud of metamaterial sheet3_108.csv",
        "Point cloud of metamaterial sheet4_145.csv",
        "Point cloud of metamaterial sheet42_145.csv",
    ]
    #        "Point cloud of metamaterial sheet_Image 52_2D.csv",
    saved_comparison_file_names = [
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
    #        "EXP_96_plane",
    associated_reference_filenames = [
        "SIM_96_REF",
        "SIM_96_REF",
        "SIM_96_REF",
        "SIM_96_REF",
        "SIM_600_REF",
        "SIM_600_REF",
        "SIM_600_REF",
        "EXP_96_REF",
        "EXP_96_REF",
        "EXP_96_REF",
        "EXP_96_REF",
        "EXP_96_REF",
        "EXP_96_REF",
        "EXP_96_REF",
    ]
    is_comparison_unity = [
        True,
        True,
        True,
        True,
        True,
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    is_comparison_fixed = [
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        True,
    ]

    use_default_radius = True
    # The factor by which the maximum minimum distance is scaled
    # when calculating the radius.
    radius_scale = 1.25

    # If default radius is not used, then set the radius used.
    manual_radius = 56

    # Defines how many hexagons a point can be contained within to be considered
    # an edge point
    hexagon_edge_number_classification = 1
    UNITY_FRAME_ROTATION_ANGLE = -30
    HEXAGON_FUDGE_FACTOR = 0.1

    preprocess(
        reference_files,
        is_reference_unity,
        saved_reference_names,
        comparison_files,
        saved_comparison_file_names,
        associated_reference_filenames,
        is_comparison_unity,
        is_comparison_fixed,
        use_default_radius,
        radius_scale,
        manual_radius,
        hexagon_edge_number_classification,
        UNITY_FRAME_ROTATION_ANGLE,
        HEXAGON_FUDGE_FACTOR,
    )
