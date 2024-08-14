To run first install dependencies in Anaconda:
numpy
open3d
warnings
scipy
os
math
pathlib
re
plotly (graph objects and figure factory)
time
imageio
shapely
matplotlib
sklearn
json



Then to run the program:
First open preprocess_files. This will preprocess the data into a standard json format for analysis and extracts hexagons for future analysis.
This should just run without any settings modifications.




Next, analyze the data by running process_data
The settings are all at the bottom.
pick the files you are interested in analyzing in the list files_to_analyze at the bottom of the script.
choose the derivative order and neighborhood size.
Pick a filename and note it for future use in a separate script (obviously this could be done better).
Run, this will process the data and save the results in a json file.

Now interpolate the data for plotting.
again the settings are at the bottom.
Select which files to process in the files_to_process list.
Select which variables to process in variables_to_process.

The relevant variables are 
"gaussian_curvature",
"mean_curvature",

You can also select
"normalized_gaussian_curvature" : normalized by characteristic length of unit cell.
"hexagon_gaussian_curvature" : gaussian curvature using hexagon centers as points instead of triangles.
"normalized_mean_curvature" : normalized by characteristic length of unit cell.

select the interpolation type: recomended "cubic"
set a interpolation destination folder name
select the data folder from the previous script.
Run.




Next, plot the data in plot_data
do_plot_3d: also plot a 3d plot in an html file (if applicable)
variables_to_plot: select what variables to plot.
same_bounds_on_same_plot_type: all plots of this type will have same upper and lower limits.
select the folder that the previous interp was saved.
choose the folder name where the plots are saved.
Run.

Now you can access the plots in the plots folder.
