import numpy as np
from warnings import warn
from time import time
from sklearn.neighbors import NearestNeighbors


def check_symmetric(a, tol=1e-8):
    return np.all(np.abs(a - a.T) < tol)


def relate_value_to_radius_and_theta_given_xy_coordinates(coordinate_set, value):
    value_set = []
    for coordinate, value in zip(coordinate_set, value):
        x = coordinate[0]
        y = coordinate[1]
        r = np.sqrt(x ** 2 + y ** 2)
        theta = np.arctan2(x, y)
        value_set.append([r, theta, value])


def dist_array(point_cloud):
    """
    Calculates the euclidean x,y (ignoring z) distance between each point in a
    point_cloud

    Parameters:
        point_cloud (np array): 3d positions of points x,y,z are columns, point
        indexes are rows.

    Returns:
        dist(np array): first point index is row, second point index to compare
        to is the column. Values are the euclidean x,y distance between the points.
        Should be symetric.
    """
    # TODO, remove need for this function, will be slow for large data sets. low
    # Priority until larger data needed.

    # Creates a list of distances from each point to each point.
    dist = np.zeros([len(point_cloud), len(point_cloud)])
    for idx, point in enumerate(point_cloud[:, 0:2]):
        # start = time()
        for comp_idx, other_point in enumerate(point_cloud[:, 0:2]):

            dist[idx, comp_idx] = np.linalg.norm(point - other_point)
        # end = time()
        # print("Loop Time: " + str(end - start))
    # dist: The row index is the first point, the column index is the
    # comparision point. Value is the distance between the points. Should be symetric.
    if not check_symmetric(dist):
        raise ValueError("Distance array is not symetric for some reason")

    return dist


def n_closest_points(X, n):
    nbrs = NearestNeighbors(n_neighbors=n + 1, algorithm="auto").fit(X)
    distances, indices = nbrs.kneighbors(X)
    chopped_distances = distances[:, 1:]
    chopped_indices = indices[:, 1:]
    return chopped_indices, chopped_distances


def closest_points_array(point_cloud, number_of_points, dist=None):
    """
    Finds the n (number of points) closest points to every point
    Parameters:
        point_cloud (array): 3d positions of points x,y,z are columns, point
        indexes are rows.
        dist (array): Array of points indicating distance between any two points
        row and columns are the point indexes, values are distances.
        number_of_points (int): number of closest points returned. If -1, then
        all points are returned

    Returns:
        closest_points (array): Array for all points where row is point index,
        and row values are the n closest points indexes (from point_cloud)
        ordered by distance excluding the index point.
    """
    #

    if dist is None:
        dist = dist_array(point_cloud)
    if number_of_points > len(point_cloud) - 1:
        number_of_points = -1
        warn(
            "Number of specified points exceeds the total number of points. Will continue with all points."
        )

    if number_of_points == -1:
        closest_pts = np.zeros([len(point_cloud), len(point_cloud) - 1])
        main_final_index = -1  # Intentional, for clarity
        sorted_final_index = -1
    elif number_of_points > 0 and isinstance(number_of_points, int):
        closest_pts = np.zeros([len(point_cloud), number_of_points])
        main_final_index = number_of_points
        sorted_final_index = number_of_points + 1
    else:
        raise ValueError("The number of points must be a positive integer or -1")
    for idx, distances in enumerate(dist):
        sorted_idxs = np.argsort(distances)
        closest_pts[idx, 0:main_final_index] = sorted_idxs[1:sorted_final_index].astype(
            int
        )
        closest_pts = closest_pts.astype(int)

    return closest_pts


# def distance(points1, points2):
#     # Points in xyz format
#     dif = points_2-points_1
#     distance_vector =
#     return distance_vector
def generate_hex_mesh(point_cloud, hexagons):
    # TODO, remove dist to increase performance (will be problem for large data sets).
    dist = dist_array(point_cloud)
    p = point_cloud
    hexagon_mesh = []
    for hexagon in hexagons:
        reference_vertex = hexagon[0]
        hexagon = np.array(hexagon)
        # print(dist[reference_vertex, hexagon])
        current_hexagon = hexagon[dist[reference_vertex, hexagon].argsort()].tolist()
        line1 = [reference_vertex, current_hexagon[3]]
        line2 = [reference_vertex, current_hexagon[4]]
        line3 = [current_hexagon[3], current_hexagon[4]]
        polygon3 = line3 + [current_hexagon[5]]
        polygon4 = [reference_vertex, current_hexagon[3], current_hexagon[4]]
        # Eval dist from points onto line1
        s = point_cloud[line1[0]] - point_cloud[line1[1]]

        M1 = point_cloud[current_hexagon[1]] - point_cloud[line1[0]]
        d1 = np.linalg.norm(np.cross(M1, s)) / np.linalg.norm(s)

        M2 = point_cloud[current_hexagon[2]] - point_cloud[line1[0]]
        d2 = np.linalg.norm(np.cross(M2, s)) / np.linalg.norm(s)

        if d1 < d2:
            polygon1 = line1 + [current_hexagon[1]]
            polygon2 = line2 + [current_hexagon[2]]
        else:
            polygon1 = line1 + [current_hexagon[2]]
            polygon2 = line2 + [current_hexagon[1]]
        hexagon_mesh.append([polygon1, polygon2, polygon3, polygon4])

    return hexagon_mesh
