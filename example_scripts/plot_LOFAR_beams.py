# %%
import os

os.chdir("..")

from beam_errors.visualization import plot_spatial_beam, plot_spectrotemporal_beam
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

# Cas and Cyg
time = "2024-07-04T19:23:00"
cas_coordinates = station.radec_to_ENU(
    right_ascension=350.8575, declination=58.148167, time=time
)  # Right Ascension 23h 23m 25.8s, Declination +58º 8' 53.4''
cyg_coordinates = station.radec_to_ENU(
    right_ascension=16.135, declination=40.733889, time=time
)  # Right Ascension 19h 59m 28.4s, Declination +40° 44' 2.1''


# %%

# Element beam
plot_spatial_beam(
    station,
    n_altitude=250,
    n_azimuth=500,
    frequency=150e6,
    vmin=-50,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="element",
    cmap="jet",
    points_of_interest=[cas_coordinates, cyg_coordinates],
)

# Tile beam (array factor only)
plot_spatial_beam(
    station,
    n_altitude=250,
    n_azimuth=500,
    frequency=150e6,
    vmin=-50,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="tile",
    cmap="jet",
    points_of_interest=[cas_coordinates, cyg_coordinates],
)

# Station beam (array factor only)
plot_spatial_beam(
    station,
    n_altitude=250,
    n_azimuth=500,
    frequency=150e6,
    vmin=-50,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="station",
    cmap="jet",
    points_of_interest=[cas_coordinates, cyg_coordinates],
)

# Station and Tile beam (array factor only)
plot_spatial_beam(
    station,
    n_altitude=250,
    n_azimuth=500,
    frequency=150e6,
    vmin=-50,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="array_factor",
    cmap="jet",
    points_of_interest=[cas_coordinates, cyg_coordinates],
)

# Full beam
plot_spatial_beam(
    station,
    n_altitude=250,
    n_azimuth=500,
    frequency=150e6,
    vmin=-50,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="full",
    cmap="jet",
    points_of_interest=[cas_coordinates, cyg_coordinates],
)

# %%

# Element beam
plot_spectrotemporal_beam(
    station,
    right_ascension=350.8575,
    declination=58.148167,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="element",
    vmin=None,
    number_of_timeslots=1800,
)

# Tile beam (array factor only)
plot_spectrotemporal_beam(
    station,
    right_ascension=350.8575,
    declination=58.148167,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="tile",
    vmin=None,
    number_of_timeslots=1800,
)


# Station beam (array factor only)
plot_spectrotemporal_beam(
    station,
    right_ascension=350.8575,
    declination=58.148167,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="station",
    vmin=None,
    number_of_timeslots=1800,
)

# Station and Tile beam (array factor only)
plot_spectrotemporal_beam(
    station,
    right_ascension=350.8575,
    declination=58.148167,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="array_factor",
    vmin=None,
    number_of_timeslots=1800,
)

# Full beam
plot_spectrotemporal_beam(
    station,
    right_ascension=350.8575,
    declination=58.148167,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="full",
    vmin=None,
    number_of_timeslots=1800,
)

# %%
