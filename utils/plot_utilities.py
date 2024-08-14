import math
from math import nan
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objs as go
import plotly.figure_factory as ff
from utils import utility as util
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

COLORSCALE = "jet"


def plot_color_gradients(cmap_list):
    # Create figure and adjust figure height to number of colormaps
    gradient = np.linspace(0, 1, 256)
    gradient = np.vstack((gradient, gradient))
    nrows = len(cmap_list)
    figh = 0.35 + 0.15 + (nrows + (nrows - 1) * 0.1) * 0.22
    fig, axs = plt.subplots(nrows=nrows + 1, figsize=(6.4, figh))

    for ax, cmap in zip(axs, cmap_list):
        ax.imshow(gradient, aspect="auto", cmap=cmap)
        ax.text(
            -0.01,
            0.5,
            cmap_list,
            va="center",
            ha="right",
            fontsize=10,
            transform=ax.transAxes,
        )

    # Turn off *all* ticks & spines, not just the ones with colormaps.
    for ax in axs:
        ax.set_axis_off()


def mpl_to_plotly(cmap, pl_entries=30, rdigits=2):
    # cmap - colormap
    # pl_entries - int = number of Plotly colorscale entries
    # rdigits - int -=number of digits for rounding scale values
    scale = np.linspace(0, 1, pl_entries)
    colors = (cmap(scale)[:, :3] * 255).astype(np.uint8)
    pl_colorscale = [
        [round(s, rdigits), f"rgb{tuple(color)}"] for s, color in zip(scale, colors)
    ]
    return pl_colorscale


def asymmetric_colormap(
    minimum, center, maximum, default_colormap="jet", divisions=256
):

    jet = cm.get_cmap(default_colormap, divisions)
    newcolors = jet(np.linspace(0, 1, divisions))
    diff = maximum - minimum
    max_perc = (maximum - center) / diff
    min_perc = 1 - max_perc

    # center_idx = math.round(min_perc * divisions)
    lower_scale_factor = min_perc / 0.5
    upper_scale_factor = max_perc / 0.5
    lower_values = np.array([*range(0, int(divisions / 2) - 1)]) / divisions
    upper_values = np.array([*range(int(divisions / 2), divisions)]) / divisions

    lower_value_scale = (lower_values) * lower_scale_factor
    upper_value_scale = (upper_values - 0.5) * upper_scale_factor + lower_value_scale[
        -1
    ]
    value_scale = np.hstack((lower_value_scale, upper_value_scale))

    colormap = []
    for idx, (value, color) in enumerate(zip(value_scale, newcolors)):
        r, g, b, a = color
        if idx == divisions - 2:
            value = 1
        colormap.append([float(value), (r, g, b, a)])
    new_cmap = LinearSegmentedColormap.from_list("temp", colormap, divisions)
    new_colormap = mpl_to_plotly(new_cmap)
    # new_map = new_cmap(np.linspace(0, 1, divisions))

    # new_colormap = []
    # new_values = np.array([*range(0, divisions)]) / divisions
    # for idx, (value, color) in enumerate(zip(new_values, new_map)):
    #     r, g, b, a = color
    #
    #     if idx == divisions - 1:
    #         value = 1
    #     new_colormap.append([float(value), f"rgba({r},{g},{b},{a})"])
    # plot_color_gradients([new_cmap])
    return new_colormap


def standard_heatmap_plot(variables, points, bounds, axis_range, plot_options, null):
    global COLORSCALE
    colorscale = COLORSCALE
    x, y, data = variables
    x_data, y_data, data_points = points

    title = plot_options["title"]
    x_label = plot_options["x_label"]
    y_label = plot_options["y_label"]
    color_title = plot_options["color_title"]

    try:
        is_asymmetric = plot_options["asymmetric_colormap"]
    except KeyError:
        is_asymmetric = False

    min_bound, max_bound = bounds
    if is_asymmetric:
        colorscale = asymmetric_colormap(min_bound, 0, max_bound, colorscale,)

    fig = go.Figure(
        go.Heatmap(
            x=x,
            y=y,
            z=data,
            zmin=min_bound,
            zmax=max_bound,
            colorbar={"title": color_title, "exponentformat": "e"},
            colorscale=colorscale,
        )
    )

    if not len(x_data) == 0:
        points = np.array(points).T
        data_points = np.array(data_points)
        is_nan_mask = np.isnan(data_points)
        out_x, out_y = points[is_nan_mask, :2].T
        in_x, in_y = points[np.logical_not(is_nan_mask), :2].T
        fig.add_trace(go.Scatter(x=out_x, y=out_y, mode="markers", marker_color="red"))
        fig.add_trace(
            go.Scatter(
                x=in_x,
                y=in_y,
                mode="markers",
                marker_symbol="circle-open",
                marker_color="black",
            )
        )
        fig.update(layout_showlegend=False)

    fig.update_layout(
        autosize=False,
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        plot_bgcolor="rgb(255, 255, 255)",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, showgrid=False, visible=False)
    fig.update_xaxes(showgrid=False, visible=False)
    fig.update_scenes(xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False)
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    fig.update_xaxes(range=axis_range)
    fig.update_yaxes(range=axis_range)
    return fig


def standard_3d_heatmap_plot(variables, points, bounds, plot_options, null):
    global COLORSCALE
    colorscale = COLORSCALE
    x, y, z, data = variables
    x_data, y_data, z_data, data_points = points

    title = plot_options["title"]
    x_label = plot_options["x_label"]
    y_label = plot_options["y_label"]
    color_title = plot_options["color_title"]
    try:
        is_asymmetric = plot_options["asymmetric_colormap"]
    except KeyError:
        is_asymmetric = False
    min_bound, max_bound = bounds
    if is_asymmetric:
        colorscale = asymmetric_colormap(min_bound, 0, max_bound, colorscale,)

    fig = go.Figure(
        data=[
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=data,
                cmin=min_bound,
                cmax=max_bound,
                colorbar={"title": color_title, "exponentformat": "e"},
                colorscale=colorscale,
            )
        ]
    )

    # if not len(x_data) == 0:
    #     fig.add_trace(go.Scatter3d(x=x_data, y=y_data, z=z_data,))

    fig.update_layout(title=title, autosize=False)
    fig.update_layout(scene_aspectmode="data")
    fig.update_scenes(xaxis_showgrid=False, yaxis_showgrid=False, zaxis_showgrid=False)
    fig.update_scenes(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False)
    return fig


def anisotropy_quiver_plot(tail_loc, data, scale, arrow_scale, width, color):
    global COLORSCALE
    colorscale = COLORSCALE
    tail_loc = np.array(tail_loc)
    data = np.array(data)
    fig = ff.create_quiver(
        tail_loc[:, 0],
        tail_loc[:, 1],
        data[:, 0],
        data[:, 1],
        scale=scale,
        arrow_scale=arrow_scale,
        name="quiver",
        line=dict(width=width, color=color),
    )
    return fig


def anisotropy_plot_function(
    variables, points, bounds, axis_range, plot_options, quivers_plot
):
    global COLORSCALE
    colorscale = COLORSCALE
    x, y, data = variables
    min_bound, max_bound = bounds
    title = plot_options["title"]
    x_label = plot_options["x_label"]
    y_label = plot_options["y_label"]
    color_title = plot_options["color_title"]
    qo = plot_options["quiver_options"]
    quiver_options_lower = [
        qo["scale"],
        qo["arrow_scale"],
        qo["first_quiver_width"],
        qo["first_color"],
    ]
    quiver_options_upper = [
        qo["scale"],
        qo["arrow_scale"],
        qo["second_quiver_width"],
        qo["second_color"],
    ]

    tail_loc = quivers_plot[0]
    quiver = quivers_plot[1]

    figures = []
    fig = anisotropy_quiver_plot(tail_loc, quiver, *quiver_options_lower)
    figures.append(fig)

    fig = anisotropy_quiver_plot(tail_loc, quiver, *quiver_options_upper)
    figures.append(fig)
    fig = anisotropy_quiver_plot(tail_loc, -quiver, *quiver_options_upper)
    figures.append(fig)
    fig = anisotropy_quiver_plot(tail_loc, -quiver, *quiver_options_lower)
    figures.append(fig)

    fig1 = figures.pop()
    for figure in figures:
        fig1.add_traces(data=figure.data)
    fig1.add_trace(
        go.Heatmap(
            x=x,
            y=y,
            z=data,
            zmin=min_bound,
            zmax=max_bound,
            colorbar={"title": color_title, "exponentformat": "e"},
            colorscale=colorscale,
        )
    )
    fig1.update_layout(
        autosize=False,
        title=title,
        showlegend=False,
        xaxis_title=x_label,
        yaxis_title=y_label,
    )
    fig1.update_xaxes(range=axis_range)
    fig1.update_yaxes(range=axis_range)
    fig1.update_yaxes(
        scaleanchor="x", scaleratio=1,
    )
    return fig1


def non_gridded_interpolator(xyz_data, interpolation_type, size, border_frac):

    numrows, numcols = size

    filtered_xyz_data = [i for i in xyz_data if not np.isnan(i).any()]

    filtered_xyz_data = np.array(filtered_xyz_data)

    try:
        xvals, yvals, data = filtered_xyz_data.T
    except ValueError:
        xvals, yvals, data = filtered_xyz_data
    # xvals, yvals, data = xyz_data.T

    data_location = (xvals, yvals)
    x_border = (xvals.max() - xvals.min()) * border_frac
    y_border = (yvals.max() - yvals.min()) * border_frac

    x_rulings = np.linspace(xvals.min() - x_border, xvals.max() + x_border, numcols)
    y_rulings = np.linspace(yvals.min() - y_border, yvals.max() + y_border, numrows)
    xi, yi = np.meshgrid(x_rulings, y_rulings)

    # Use voronoi diagram to color curvature of each point
    z_interpolated = griddata(data_location, data, (xi, yi), method=interpolation_type)

    # This generates Nans outside the interplation area, required since nearest
    # doesn't consider Nan a point.
    if interpolation_type == "nearest":
        z_bound_generator = griddata(data_location, data, (xi, yi), method="linear")
        z_bound = z_bound_generator * 0
        z_interpolated = z_interpolated + z_bound

    return x_rulings, y_rulings, z_interpolated


def set_bounds(data, lower=nan, center=nan, upper=nan):
    # data = [item for items in data for item in items]

    min_temp = np.nanmin(data)
    max_temp = np.nanmax(data)
    lower_dict = {"min": min_temp}
    center_dict = {}
    upper_dict = {"max": max_temp}

    if isinstance(lower, str):
        try:
            lower = lower_dict[lower]
        except KeyError:
            raise ValueError("The lower arguement is not defined in function")
    if isinstance(center, str):
        try:
            center = center_dict[center]
        except KeyError:
            raise ValueError("The center arguement is not defined in function")
    if isinstance(upper, str):
        try:
            upper = upper_dict[upper]
        except KeyError:
            raise ValueError("The upper is not defined in function")

    if lower > upper:
        raise ValueError("Upper bound must be larger than lower bound")

    if not math.isnan(center):
        if math.isnan(lower) and math.isnan(upper):
            if abs(min_temp - center) > abs(max_temp - center):
                min_bound = min_temp
                max_bound = -(min_temp - center) + center

            else:
                min_bound = -(max_temp - center) + center
                max_bound = max_temp
        elif not math.isnan(lower):
            min_bound = lower
            max_bound = -(lower - center) + center
        elif not math.isnan(upper):
            min_bound = -(upper - center) + center
            max_bound = upper
        else:
            max = (abs(min_temp - center), abs(max_temp - center))
            min_bound = -abs(max - center) + center
            max_bound = abs(max - center) + center
    else:
        min_bound = min_temp
        max_bound = max_temp
        if not math.isnan(upper):
            max_bound = upper
        if not math.isnan(lower):
            min_bound = lower
    return [min_bound, max_bound]


def get_bounds_all_plots(files, variable_list, INTERP_FOLDER, PLOT_FUNCTION_DICT):
    data = []
    bounds = []
    for file in files:
        for variable in variable_list:
            data = data + util.get_variables_from_json(
                INTERP_FOLDER + file + "." + variable, "data_grid"
            )
    bound_settings = PLOT_FUNCTION_DICT[variable_list[0]][0]["bound_type"]
    bound = set_bounds(data, *bound_settings)
    for i in range(len(files) * len(variable_list)):
        bounds.append(bound)
    return bounds


def get_bounds_same_file(files, variable_list, INTERP_FOLDER, PLOT_FUNCTION_DICT):
    bounds = []
    for file in files:
        data = []
        for variable in variable_list:
            data = data + util.get_variables_from_json(
                INTERP_FOLDER + file + "." + variable, "data_grid"
            )
        bound_settings = PLOT_FUNCTION_DICT[variable_list[0]][0]["bound_type"]
        bound = set_bounds(data, *bound_settings)
        for i in variable_list:
            bounds.append(bound)
    return bounds


def axis_range_gen(x, y):
    xmin, xmax, ymin, ymax = [min(x), max(x), min(y), max(y)]
    max_pix_range = 500
    # max([(ymax - ymin),(xmax - xmin)])
    height_data = max_pix_range
    width_data = max_pix_range
    max_range = max([ymax, xmax])
    min_range = min([ymin, xmin])
    axis_range = [min_range, max_range]
    return axis_range
