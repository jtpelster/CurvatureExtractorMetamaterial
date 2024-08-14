from utils import plot_utilities as putil
from utils.plot_utilities import standard_heatmap_plot

from utils.plot_utilities import standard_3d_heatmap_plot
from utils.plot_utilities import anisotropy_plot_function


from utils import utility as util
from math import nan
import numpy as np
from pathlib import Path
import os
import plotly
import re
import plotly.graph_objs as go
import plotly.figure_factory as ff
import math
from time import time
from time import sleep
import imageio


def standard_3d_heatmap_plot_temp(variables, points, bounds, plot_options, null):
    global COLORSCALE
    x, y, z, data = variables
    x_data, y_data, z_data, data_points = points

    x_data = x_data - np.mean(x_data)
    y_data = y_data - np.mean(y_data)
    #
    z_data = z_data - np.mean(z_data)

    title = plot_options["title"]
    x_label = plot_options["x_label"]
    y_label = plot_options["y_label"]
    color_title = plot_options["color_title"]
    min_bound, max_bound = bounds

    angles = np.linspace(0, 2 * math.pi, 10)

    for angle in angles:
        R2 = np.array(
            [
                [1, 0, 0],
                [0, math.cos(angle), math.sin(angle)],
                [
                    0,
                    -math.sin(angle),
                    math.cos(angle),
                ],
            ]
        )

        x1, y1, z1 = R2 @ np.array([x_data, y_data, z_data])

        # x1 = np.reshape(x1, np.shape(x_m))
        # y1 = y1.reshape(np.shape(y_m))
        # z1 = z1.reshape(np.shape(z))

        x2, y2, z2 = putil.non_gridded_interpolator(
            np.array([x1, y1, z1]), "linear", [200, 200], 0.02
        )
        null, null, data = putil.non_gridded_interpolator(
            np.array([x1, y1, data_points]).T, "cubic", [200, 200], 0.02
        )
        # print(np.shape(x1))
        # print(x1)
        # print(y1)
        # print(z1)

        # for x,y,z zip(np.flatten())
        # x1, null, null = R2 @ np.array(
        #     np.vstack((np.array([x]), np.zeros((2, np.shape(x)))))
        # )
        # null, y1, null = R2 @ np.array(
        #     np.vstack((np.zeros((1, (y))), np.array([y]), np.zeros((1, len(y)))))
        # )
        # z_r = np.ravel(z)
        # null, null, z_rot = R2 @ np.array(
        #     np.vstack((np.zeros((2, len(z_r))), np.array([z_r])))
        # )
        # z1 = z_rot.reshape(np.shape(z))

        # x1, y1, z1 = R2 @ np.array([xa, ya, za])
        # x1.reshape(np.shape(x))
        # y1.reshape(np.shape(y))
        # z1.reshape(np.shape(z))
        # x1=

        # x2 = [x for x in x1 if not np.isnan(x)]
        # y2 = [y for y in y1 if not np.isnan(y)]
        # z2 = [z for z in z1 if not np.isnan(z)]

        # data_ravel = np.ravel(data)
        # data_trim = [d for da, x in zip(data_ravel, x1) for d in da if not np.isnan(x)]
        # print(len(x2))
        # print(len(y2))
        # print(len(z2))
        # print(len(data_trim))
        # mask = 0 * x1
        # data_trim = mask * data
        fig = go.Figure(
            data=[
                go.Surface(
                    x=x2,
                    y=y2,
                    z=z2,
                    surfacecolor=data,
                    cmin=min_bound,
                    cmax=max_bound,
                    colorbar={"title": color_title, "exponentformat": "e"},
                    colorscale=COLORSCALE,
                )
            ]
        )

        fig.update_layout(title=title, autosize=False)
        fig.update_layout(scene_aspectmode="data")
        fig.update_scenes(
            xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False
        )
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
        fig.show()

        sleep(2)
    return fig


def save_gif(folder, saving_folder, name):
    images = []
    filenames = util.get_files(folder, ".jpg")
    print(filenames)
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(saving_folder + name + ".gif", images)

    writer = imageio.get_writer(saving_folder + name + ".mp4", fps=5)

    for im in filenames:
        writer.append_data(imageio.imread(im))
    writer.close()
    pass


def delete_images(folder):
    filenames = util.get_files(folder, ".jpg")
    for filename in filenames:
        os.remove(filename)
    pass


def animate(figure, name):
    gif_dir = ".\\gif_dir\\"
    temp_dir = ".\\temp_dir\\"
    if not os.path.isdir(gif_dir):
        os.makedirs(gif_dir)
    if not os.path.isdir(temp_dir):
        os.makedirs(temp_dir)
    angles = np.linspace(0, 2 * math.pi)
    for idx, angle in enumerate(angles):
        base_angle = 0
        R1 = np.array(
            [
                [1, 0, 0],
                [0, math.cos(base_angle), math.sin(base_angle)],
                [
                    0,
                    -math.sin(base_angle),
                    math.cos(base_angle),
                ],
            ]
        )
        R2 = np.array(
            [
                [1, 0, 0],
                [0, math.cos(angle), math.sin(angle)],
                [
                    0,
                    -math.sin(angle),
                    math.cos(angle),
                ],
            ]
        )
        starting_xy_angle = np.array([0, 0, 1] / np.linalg.norm([0, 0, 1]))
        x, y, z = R2 @ R1 @ starting_xy_angle
        print((x, y, z))
        camera = dict(up=dict(x=1, y=0, z=0.5), eye=dict(x=3 * x, y=3 * y, z=3 * z))
        figure.update_layout(scene_camera=camera, scene_dragmode="orbit")
        figure.write_image(
            temp_dir + "figure_" + f"{idx:03d}" + ".jpg", engine="kaleido"
        )
    save_gif(temp_dir, gif_dir, name)
    delete_images(temp_dir)


# same_bounds_on_same_plot_type = [True,True]
quiver_options = {
    "scale": 40,
    "arrow_scale": 0.01,
    "first_quiver_width": 3,
    "second_quiver_width": 2,
    "first_color": "#000000",
    "second_color": "#00FF7F",
}

isotropic_gaussian_curvature = {
    "function": standard_heatmap_plot,
    "bound_type": [nan, 0, nan],
    "quiver": False,
    "file_title": "isotropic_gaussian_curvature",
    "is_3d": False,
    "data_points": True,
    "do_animation": False,
    "plot_options": {
        "title": "Isotropic Gaussian Curvature Calculation",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Curvature<br>(1/\u03bcm^2)",
        "asymmetric_colormap": False,
    },
}
isotropic_gaussian_curvature_3d = {
    "function": standard_3d_heatmap_plot,
    "bound_type": [nan, 0, nan],
    "quiver": False,
    "is_3d": True,
    "file_title": "isotropic_gaussian_curvature_3d",
    "data_points": False,
    "do_animation": False,
    "plot_options": {
        "title": "Isotropic Gaussian Curvature Calculation",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Curvature<br>(1/\u03bcm^2)",
        "asymmetric_colormap": False,
    },
}
normalized_isotropic_gaussian_curvature = {
    "function": standard_heatmap_plot,
    "bound_type": [nan, 0, nan],
    "quiver": False,
    "is_3d": False,
    "file_title": "normalized_isotropic_gaussian_curvature",
    "data_points": True,
    "do_animation": False,
    "plot_options": {
        "title": "Normalized Isotropic Gaussian Curvature",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Curvature<br>sqrt(K*L^2)",
        "asymmetric_colormap": False,
    },
}

gaussian_curvature = {
    "function": standard_heatmap_plot,
    "bound_type": ["min", nan, "max"],
    "quiver": False,
    "is_3d": False,
    "file_title": "gaussian_curvature",
    "data_points": True,
    "do_animation": False,
    "plot_options": {
        "title": "Gaussian Curvature",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Curvature<br>(1/\u03bcm^2)",
        "data_points": True,
        "asymmetric_colormap": False,
    },
}
gaussian_curvature_3d = {
    "function": standard_3d_heatmap_plot,
    "bound_type": ["min", nan, "max"],
    "quiver": False,
    "is_3d": True,
    "file_title": "gaussian_curvature_3d",
    "data_points": True,
    "do_animation": False,
    "plot_options": {
        "title": "Gaussian Curvature",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Curvature<br>(1/\u03bcm^2)",
        "asymmetric_colormap": False,
    },
}
normalized_gaussian_curvature = {
    "function": standard_heatmap_plot,
    "bound_type": [nan, 0, nan],
    "quiver": False,
    "is_3d": False,
    "file_title": "normalized_gaussian_curvature",
    "data_points": True,
    "do_animation": False,
    "plot_options": {
        "title": "Normalized Gaussian Curvature",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Curvature<br>sqrt(K*l^2)",
        "asymmetric_colormap": False,
    },
}
normalized_gaussian_curvature_3d = {
    "function": standard_3d_heatmap_plot,
    "bound_type": [nan, 0, nan],
    "quiver": False,
    "is_3d": True,
    "file_title": "normalized_gaussian_curvature_3d",
    "data_points": False,
    "do_animation": False,
    "plot_options": {
        "title": "Normalized Gaussian Curvature",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Normalized Curvature<br>(sqrt(K)*L)",
        "asymmetric_colormap": False,
    },
}
hexagon_gaussian_curvature = {
    "function": standard_heatmap_plot,
    "bound_type": [nan, 0, nan],
    "quiver": False,
    "is_3d": False,
    "file_title": "hexagon_gaussian_curvature",
    "data_points": False,
    "do_animation": False,
    "plot_options": {
        "title": "Gaussian Curvature",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Curvature<br>(1/\u03bcm^2)",
        "asymmetric_colormap": False,
    },
}

mean_curvature = {
    "function": standard_heatmap_plot,
    "bound_type": ["min", nan, "max"],
    "quiver": False,
    "is_3d": False,
    "file_title": "mean_curvature",
    "data_points": True,
    "do_animation": False,
    "plot_options": {
        "title": "Mean Curvature",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Curvature<br>(1/\u03bcm)",
        "asymmetric_colormap": False,
    },
}
normalized_mean_curvature = {
    "function": standard_heatmap_plot,
    "bound_type": [nan, 0, nan],
    "quiver": False,
    "is_3d": False,
    "file_title": "normalized_mean_curvature",
    "data_points": True,
    "do_animation": False,
    "plot_options": {
        "title": "Normalized Mean Curvature",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Normalized Curvature<br>(H*L)",
        "asymmetric_colormap": True,
    },
}
mean_curvature_3d = {
    "function": standard_3d_heatmap_plot,
    "bound_type": ["min", nan, "max"],
    "quiver": False,
    "is_3d": True,
    "file_title": "mean_curvature_3d",
    "data_points": True,
    "do_animation": False,
    "plot_options": {
        "title": "Mean Curvature",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Normalized Curvature<br>(H*L)",
        "asymmetric_colormap": True,
    },
}
normalized_mean_curvature_3d = {
    "function": standard_3d_heatmap_plot,
    "bound_type": [nan, 0, nan],
    "quiver": False,
    "is_3d": True,
    "file_title": "normalized_mean_curvature_3d",
    "data_points": False,
    "do_animation": False,
    "plot_options": {
        "title": "Normalized Mean Curvature",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Curvature<br>(1/\u03bcm)",
        "asymmetric_colormap": False,
    },
}
curv_rmse = {
    "function": standard_heatmap_plot,
    "bound_type": [0, nan, nan],
    "quiver": False,
    "is_3d": False,
    "file_title": "curv_rmse",
    "data_points": True,
    "do_animation": False,
    "plot_options": {
        "title": "Curvature Derivative RMSE",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "(\u03bcm)",
        "asymmetric_colormap": False,
    },
}

relative_anisotropy = {
    "function": anisotropy_plot_function,
    "bound_type": [0, nan, nan],
    "quiver": True,
    "is_3d": False,
    "file_title": "relative_anisotropy",
    "data_points": False,
    "do_animation": False,
    "plot_options": {
        "title": "Relative Anisotropy",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Anisotropy",
        "quiver_options": quiver_options,
        "asymmetric_colormap": False,
    },
}
anisotropy = {
    "function": anisotropy_plot_function,
    "bound_type": [1, nan, nan],
    "quiver": True,
    "is_3d": False,
    "file_title": "anisotropy",
    "data_points": False,
    "do_animation": False,
    "plot_options": {
        "title": "Anisotropy",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Anisotropy",
        "quiver_options": quiver_options,
        "asymmetric_colormap": False,
    },
}
area_expansion = {
    "function": standard_heatmap_plot,
    "bound_type": [nan, 1, nan],
    "quiver": False,
    "is_3d": False,
    "file_title": "area_expansion",
    "data_points": False,
    "do_animation": False,
    "plot_options": {
        "title": "Area Expansion Ratio",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Ratio",
        "asymmetric_colormap": False,
    },
}
area_expansion_3d = {
    "function": standard_3d_heatmap_plot,
    "bound_type": [nan, 1, nan],
    "quiver": False,
    "is_3d": True,
    "file_title": "area_expansion_3d",
    "data_points": False,
    "do_animation": False,
    "plot_options": {
        "title": "Area Expansion Ratio",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Ratio",
        "data_points": False,
        "asymmetric_colormap": False,
    },
}
isotropic_rmse = {
    "function": standard_heatmap_plot,
    "bound_type": [0, nan, nan],
    "quiver": False,
    "is_3d": False,
    "file_title": "isotropic_rmse",
    "data_points": False,
    "do_animation": False,
    "plot_options": {
        "title": "Isotropic Gaussian RMSE",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "(\u03bcm)",
        "asymmetric_colormap": False,
    },
}
isotropic_comparison = {
    "function": standard_heatmap_plot,
    "bound_type": [nan, 0, nan],
    "quiver": False,
    "is_3d": False,
    "file_title": "isotropic_comparison",
    "data_points": False,
    "do_animation": False,
    "plot_options": {
        "title": "Normalized Measured Gaussian Minus Isotropic Estimate",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Curvature<br>(1/\u03bcm^2)",
        "asymmetric_colormap": False,
    },
}
normalized_isotropic_comparison = {
    "function": standard_heatmap_plot,
    "bound_type": [nan, 0, nan],
    "quiver": False,
    "is_3d": False,
    "file_title": "normalized_isotropic_comparison",
    "data_points": False,
    "do_animation": False,
    "plot_options": {
        "title": "Normalized Measured Gaussian Minus Isotropic Estimate",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "difference/max_measured",
        "asymmetric_colormap": False,
    },
}
area_dist_error = {
    "function": standard_heatmap_plot,
    "bound_type": [0, nan, nan],
    "quiver": False,
    "is_3d": False,
    "file_title": "area_dist_error",
    "data_points": False,
    "do_animation": False,
    "plot_options": {
        "title": "RMSE Area Calculation Fit",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "\u03bcm",
        "asymmetric_colormap": False,
    },
}
out_of_plane_aniosotropy_variation = {
    "function": standard_heatmap_plot,
    "bound_type": [0, nan, nan],
    "quiver": False,
    "is_3d": False,
    "file_title": "out_of_plane_aniosotropy_variation",
    "data_points": False,
    "do_animation": False,
    "plot_options": {
        "title": "(Eig 3)/min(Eig 1,2)",
        "x_label": "X Position (\u03bcm)",
        "y_label": "Y Position (\u03bcm)",
        "color_title": "Eig Ratio",
        "asymmetric_colormap": False,
    },
}


PLOT_FUNCTION_DICT = {
    "isotropic_gaussian_curvature": [
        isotropic_gaussian_curvature,
        isotropic_gaussian_curvature_3d,
    ],
    "normalized_isotropic_gaussian_curvature": [
        normalized_isotropic_gaussian_curvature,
    ],
    "isotropic_rmse": [isotropic_rmse],
    "isotropic_comparison": [isotropic_comparison],
    "normalized_isotropic_comparison": [normalized_isotropic_comparison],
    "normalized_gaussian_curvature": [
        normalized_gaussian_curvature,
        normalized_gaussian_curvature_3d,
    ],
    "normalized_mean_curvature": [
        normalized_mean_curvature,
        normalized_mean_curvature_3d,
    ],
    "hexagon_gaussian_curvature": [hexagon_gaussian_curvature],
    "gaussian_curvature": [gaussian_curvature, gaussian_curvature_3d],
    "mean_curvature": [
        mean_curvature,
        mean_curvature_3d,
    ],
    "curv_rmse": [curv_rmse],
    "relative_anisotropy": [relative_anisotropy],
    "anisotropy": [anisotropy],
    "out_of_plane_aniosotropy_variation": [out_of_plane_aniosotropy_variation],
    "area_expansion": [area_expansion, area_expansion_3d],
    "area_dist_error": [area_dist_error],
}


# save_image()
# write_gif()


def plot(
    variable,
    bounds,
    file_name,
    figures,
    do_plot_3d,
    parameters,
    INTERP_FOLDER,
    DATA_FOLDER,
    PLOT_FOLDER,
):
    start = time()
    extra_plot = [nan]
    plot_options = parameters["plot_options"]
    skip_plot = 0
    print(8)
    x_data, y_data, z_data, data_points = ([], [], [], [])
    if parameters["data_points"]:
        x_data, y_data, z_data, data_points = util.get_variables_from_json(
            INTERP_FOLDER + file_name + "." + variable,
            ["x_data", "y_data", "z_data", "data_points"],
        )
    print(9)
    if parameters["is_3d"] and do_plot_3d:
        x, y, z, data = util.get_variables_from_json(
            INTERP_FOLDER + file_name + "." + variable,
            ["x_grid", "y_grid", "z", "data_grid"],
        )
        figure = parameters["function"](
            [x, y, z, data],
            [x_data, y_data, z_data, data_points],
            bounds,
            plot_options,
            extra_plot,
        )
        if parameters["do_animation"]:
            animate(figure, file_name + variable)
            # parameters["function"](
            #     [x, y, z, data],
            #     [x_data, y_data, z_data, data_points],
            #     bounds,
            #     plot_options,
            #     extra_plot,
            # )
    elif parameters["is_3d"]:
        skip_plot = 1
        print(10)

    else:
        print(10)
        x, y, data = util.get_variables_from_json(
            INTERP_FOLDER + file_name + "." + variable,
            ["x_grid", "y_grid", "data_grid"],
        )
        if parameters["quiver"]:
            tail_loc, quivers = util.get_variables_from_json(
                DATA_FOLDER + file_name, ["hex_coordinates", "eigenvectors"]
            )
            quivers = np.array(quivers)
            quiver = quivers[:, 0, :]
            extra_plot = [tail_loc, quiver]
        axis_range = putil.axis_range_gen(x, y)

        figure = parameters["function"](
            [x, y, data],
            [x_data, y_data, data_points],
            bounds,
            axis_range,
            plot_options,
            extra_plot,
        )
    print(11)
    if not skip_plot:
        print(12)
        plot_time = time()
        title = parameters["file_title"]
        new_path = PLOT_FOLDER + file_name + "/"

        header_switch = 0
        if not os.path.isdir(new_path):
            os.makedirs(new_path)
        print(16)
        if not os.path.exists(new_path + HTML_FILE_NAME + ".html"):
            header_switch = 1
        print(14)
        figure.write_html(new_path + title + ".html")
        print(15)
        header = """<h1>""" + file_name
        print(13)
        with open(new_path + HTML_FILE_NAME + ".html", "a") as f:
            print(new_path + HTML_FILE_NAME + ".html")
            if header_switch:
                f.write(header)
            figure = figure.to_html(full_html=False, include_plotlyjs="cdn")
            edited_figure = figure.replace(
                re.findall("height:100%", figure)[0], "height:60%"
            )
            f.write(edited_figure)

        with open(PLOT_FOLDER + title + ".html", "a") as g:
            g.write(header + edited_figure)

            # try:
            #     variable_figures = figures[variable]
            # except KeyError:
            #     variable_figures = []
            # print(edited_figure)
            # variable_figures.append(header+edited_figure)
            # print(variable_figures)
            # figures.update({variable:variable_figures})
            # print(figures)

        end = time()
        print("Saving: " + new_path + title + ".svg")
        print("Plot time: " + str(plot_time - start))
        print("Saving time: " + str(end - plot_time))
    pass


def delete_html():
    files = list(Path(MAIN_PLOT_FOLDER).rglob(HTML_FILE_NAME + ".html"))
    for file in files:
        os.remove(file)


def delete_variable_html(PLOT_FOLDER):
    files = list(Path(PLOT_FOLDER).glob("*.html"))
    for file in files:
        os.remove(file)


def write_variable_html(figures):
    print(figures)
    for variable in figures:
        start = time()
        if not os.path.exists(variable + ".html"):
            header = """<h1>""" + variable
        fig_plot = figures[variable]
        print(fig_plot)
        with open(variable + ".html", "a") as f:
            print(fig_plot)
            print(variable + ".html")
            for figure in fig_plot:
                f.write(figure)
        end = time()
        print("Plot Save Time " + str(end - start))


DEFAULT_COLORSCALE = "jet"
MAIN_INTERP_FOLDER = "./interpolated_data/"
INTERP_DEFAULT_FOLDER = "/interp_default/"
MAIN_DATA_FOLDER = "./processed_data/"
DATA_DEFAULT_FOLDER = "./standard_process/"
MAIN_PLOT_FOLDER = "./plots/"
PLOT_DEFAULT_FOLDER = "/Default_plot/"
HTML_FILE_NAME = "Summary"


def plot_fx(
    files_to_plot,
    variables_to_plot,
    do_plot_3d,
    same_bounds_on_same_plot_type,
    user_bound_set,
    colorscale=DEFAULT_COLORSCALE,
    INTERP_UPPER_FOLDER=INTERP_DEFAULT_FOLDER,
    DATA_UPPER_FOLDER=DATA_DEFAULT_FOLDER,
    PLOT_UPPER_FOLDER=PLOT_DEFAULT_FOLDER,
):
    print(4)
    global COLORSCALE
    COLORSCALE = colorscale

    INTERP_FOLDER = MAIN_INTERP_FOLDER + INTERP_UPPER_FOLDER
    DATA_FOLDER = MAIN_DATA_FOLDER + DATA_UPPER_FOLDER
    PLOT_FOLDER = MAIN_PLOT_FOLDER + PLOT_UPPER_FOLDER

    if len(variables_to_plot) != len(same_bounds_on_same_plot_type):
        raise ValueError(
            "Length of same_bounds_on_same_plot_type and variables_to_plot must be equal"
        )
    print(5)
    delete_html()
    print(6)
    delete_variable_html(PLOT_FOLDER)
    print(7)
    figures = {}
    print(1)
    for variables, same_bounds, user_bounds in zip(
        variables_to_plot, same_bounds_on_same_plot_type, user_bound_set
    ):
        if user_bounds == "auto":
            if same_bounds:
                bounds = putil.get_bounds_all_plots(
                    files_to_plot, variables, INTERP_FOLDER, PLOT_FUNCTION_DICT
                )
            else:
                bounds = putil.get_bounds_same_file(
                    files_to_plot, variables, INTERP_FOLDER, PLOT_FUNCTION_DICT
                )
        else:
            bounds = []
            for file in files_to_plot:
                for variable in variables:
                    bounds.append(user_bounds)
        for b_idx, file in enumerate(files_to_plot):
            print(2)
            for v_idx, variable in enumerate(variables):
                current_bound = bounds.pop(0)
                for parameters in PLOT_FUNCTION_DICT[variable]:
                    print(3)
                    plot(
                        variable,
                        current_bound,
                        file,
                        figures,
                        do_plot_3d,
                        parameters,
                        INTERP_FOLDER,
                        DATA_FOLDER,
                        PLOT_FOLDER,
                    )
    print("Plotting Complete")


if __name__ == "__main__":
    files_to_plot = [
        "SIM_96_dome",
        "SIM_96_heart",
        "SIM_96_saddle",
        "SIM_96_plane",
        "SIM_600_dome",
        "SIM_600_planar",
        "SIM_600_saddle",
        "EXP_96_saddle",
        "EXP_96_plane",
        "EXP_96_dome",
        "EXP_96_1",
        "EXP_96_2",
        "EXP_96_3",
        "EXP_96_4",
    ]
    # files_to_plot = [
    #     "EXP_96_1_12",
    #     "EXP_96_2_12",
    #     "EXP_96_3_12",
    #     "EXP_96_4_12",
    #     "EXP_96_1_18",
    #     "EXP_96_2_18",
    #     "EXP_96_3_18",
    #     "EXP_96_4_18",
    #     "EXP_96_1_24",
    #     "EXP_96_2_24",
    #     "EXP_96_3_24",
    #     "EXP_96_4_24",
    # ]

    

    # files_to_plot = [
    #     "EXP_96_saddle",
    #     "EXP_96_plane",
    #     "EXP_96_dome",
    # ]
    # "EXP_96_saddle",
    # Combine variables into a sublist that should have the same bounds
    # Single variables should be in their own sublist
    # e.g. gaussian_curvature and isotropic_gaussian_curvature might want the same
    # bound so:
    # [["isotropic_gaussian_curvature","gaussian_curvature"],["area_expansion"]]

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

    # variables_to_plot = [
    #     ["isotropic_gaussian_curvature"],
    #     ["gaussian_curvature"],
    #     ["normalized_gaussian_curvature"],
    #     ["hexagon_gaussian_curvature"],
    #     ["mean_curvature"],
    #     ["normalized_mean_curvature"],
    #     ["area_expansion"],
    #     ["relative_anisotropy"],
    #     ["anisotropy"],
    # ]

    do_plot_3d = True
    variables_to_plot = [["gaussian_curvature"], ["mean_curvature"]]

    # This option will cause all plots of the same variable to have the same bounds
    # If variables in variables_to_plot are grouped together, and this is False,
    # Then only the bounds from the same file in the variable group will match.
    #same_bounds_on_same_plot_type = [True, True, True, True, True, True]
    same_bounds_on_same_plot_type = [
        True,
        True,
    ]
    user_bounds = [
        "auto",
        "auto",
    ]
    interp_folder = "cubic_interp"
    plot_folder = "cubic_interp"
    plot_fx(
        files_to_plot,
        variables_to_plot,
        do_plot_3d,
        same_bounds_on_same_plot_type,
        user_bounds,
        INTERP_UPPER_FOLDER= interp_folder + "/",
        PLOT_UPPER_FOLDER=plot_folder + "/",
    )
# write_variable_html(figures)
