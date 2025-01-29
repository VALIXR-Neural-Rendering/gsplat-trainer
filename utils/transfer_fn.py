# Run this file to create an example DCP dataset

import argparse
from netCDF4 import Dataset
import laspy
import numpy as np
import matplotlib as plt
import xarray as xr

EPS = 1e-6


class TransferFunction:
    def __init__(
        self,
        range_min,
        range_max,
        control_vals,
        opacity=[1.0] * 4,
        colmap="bwr",
    ):
        self.range_min = range_min
        self.range_max = range_max
        self.control_vals = control_vals
        self.opacity = opacity
        self.colmap = self.get_colmap(colmap)

    def get_colmap(self, colmap, res=256):
        return plt.colormaps[colmap]

    def normalize(self, val):
        return (val - self.range_min) / (
            (self.range_max - self.range_min) + EPS
        )

    def get_color(self, vals):
        rgb = np.zeros((vals.shape[0], 3))
        rgb = self.colmap(self.normalize(vals))
        rgb[(vals < self.range_min) & (vals > self.range_max)] = np.zeros(
            rgb.shape[1]
        )
        rgb[vals < self.control_vals[0]] *= self.opacity[0]
        rgb[vals > self.control_vals[-1]] *= self.opacity[-1]
        for i in range(len(self.control_vals) - 1):
            ctrl_range = (vals >= self.control_vals[i]) & (
                vals <= self.control_vals[i + 1]
            )
            ratio = (vals - self.control_vals[i]) / (
                (self.control_vals[i + 1] - self.control_vals[i]) + EPS
            )
            alpha = min(self.opacity[i], self.opacity[i + 1]) + ratio * np.abs(
                self.opacity[i + 1] - self.opacity[i]
            )
            alpha[~ctrl_range] = 1.0
            rgb[:, -1] *= alpha
        return np.asarray(rgb * 255, dtype=np.uint8)


def colorize_particles(data, var, tf):
    if var[0] == "vel":  # visualize z-component of velocity
        field = np.squeeze(data.variables[var[0]][:, :, 2])
    else:  # visualize scalar field
        field = np.squeeze(data.variables[var[0]][:])
    print("Getting colors...")
    fcols = tf.get_color(field)

    print("Writing to file...")
    carr = xr.DataArray(np.expand_dims(fcols, axis=0), dims=("T", "id", "rgba"))
    data["color"] = carr
    print("Done")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NetCDF utilities script")
    parser.add_argument(
        "--fname",
        type=str,
        default="particles_008_hd.nc4",
        help="Name of the netcdf file",
    )

    args = parser.parse_args()

    data = Dataset(args.fname, "a", format="NETCDF4")

    # Create transfer function
    range_min = -20.0
    range_max = 20.0
    control_vals = [-7.0, -3.5, 3.5, 7.0]
    opacity = [1.0, 0, 0, 1.0]
    tf = TransferFunction(range_min, range_max, control_vals, opacity)
    viz_var = ["vel"]

    # Colorize the particles
    colorize_particles(data, viz_var, tf)

    lasfile = laspy.create(point_format=2, file_version="1.2")
    pos = data.variables["Position"][:]
    col = data.variables["color"][:]
    vis_filt = np.expand_dims((col[0, :, 3] > 0), axis=0)

    # lasfile.header.offset = offset
    lasfile.header.scale = [1.0, 1.0, 1.0]

    lasfile.x = pos[vis_filt][:, 0]
    lasfile.y = pos[vis_filt][:, 1]
    lasfile.z = pos[vis_filt][:, 2]

    lasfile.red = col[vis_filt][:, 0]
    lasfile.green = col[vis_filt][:, 1]
    lasfile.blue = col[vis_filt][:, 2]
    lasfile.intensity = col[vis_filt][:, 3]

    lasfile.write("../pts_new.las")
    data.close()
