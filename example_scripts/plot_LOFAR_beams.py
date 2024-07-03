# %%
import os

os.chdir("..")

from beam_errors.visualization import plot_beam
from beam_errors.load_array import load_LOFAR
import numpy as np

array = load_LOFAR(mode="EoR")

# Core Station (CS001HBA0)
station = array["CS002HBA0"]

# Example pointings ENU
station.update_station(new_pointing=[0, 0, 1])  # Zenith
station.update_station(new_pointing=[1 / np.sqrt(2), 0, 1 / np.sqrt(2)])  # East
station.update_station(new_pointing=[1, 1, 1] / np.sqrt(3))  # NE
station.update_station(new_pointing=[-1, -2, 1] / np.sqrt(6))  # SSW
station.update_station(new_pointing=[-1, 2, 3] / np.sqrt(14))  # NNW

# NCP pointing
rotation_matrix = station.ENU_rotation_matrix()
NCP_ENU = np.array([0, 0, 1]) @ rotation_matrix.T
station.update_station(new_pointing=NCP_ENU)


# Element beam
plot_beam(
    station,
    n_altitude=250,
    n_azimuth=500,
    frequency=150e6,
    vmin=-50,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="element",
    cmap="jet",
)

# Tile beam (array factor only)
plot_beam(
    station,
    n_altitude=250,
    n_azimuth=500,
    frequency=150e6,
    vmin=-50,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="tile",
    cmap="jet",
)

# Station beam (array factor only)
plot_beam(
    station,
    n_altitude=250,
    n_azimuth=500,
    frequency=150e6,
    vmin=-50,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="station",
    cmap="jet",
)

# Station and Tile beam (array factor only)
plot_beam(
    station,
    n_altitude=250,
    n_azimuth=500,
    frequency=150e6,
    vmin=-50,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="array_factor",
    cmap="jet",
)

# Full beam
plot_beam(
    station,
    n_altitude=250,
    n_azimuth=500,
    frequency=150e6,
    vmin=-50,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="full",
    cmap="jet",
)

# %%
