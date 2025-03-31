# %%
from shimmerr.visualization import plot_spatial_beam, plot_spectrotemporal_beam
from shimmerr.load_array import load_LOFAR, load_array_from_file
import numpy as np

array = load_LOFAR(mode="EoR")

# %%
# Core Station (CS001HBA0)
array["CS002HBA0"].elements[0].elements[3].g = 3
station = array["CS002HBA0"]


# Example pointing
station.update_station_pointing(None, None)  # Drift-scan
station.update_station_pointing(new_pointing_dec=90)  # NCP pointing

# %%
# Cas and Cyg
time = "2024-07-04T19:23:00"
time = "2024-07-12T06:00:00"
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
    time=time,
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
    time=time,
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
    time=time,
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
    time=time,
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
    time=time,
    points_of_interest=[cas_coordinates, cyg_coordinates],
)

# Full beam, unit tile
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
    time=time,
    points_of_interest=[cas_coordinates, cyg_coordinates],
    calculate_all_tiles=False,
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
    utc_starttime=time,
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
    utc_starttime=time,
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
    utc_starttime=time,
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
    utc_starttime=time,
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
    utc_starttime=time,
    number_of_timeslots=1800,
)

# Full beam, unit tile
plot_spectrotemporal_beam(
    station,
    right_ascension=350.8575,
    declination=58.148167,
    antenna_mode="simplified",
    beam_plot_mode="power",
    beam_value_mode="full",
    vmin=None,
    utc_starttime=time,
    number_of_timeslots=1800,
    calculate_all_tiles=False,
)

# %%
