import numpy as np
from matplotlib import pyplot as plt


def get_beam(
    beam_value_mode,
    beam_plot_mode,
    station,
    frequency,
    directions,
    antenna_mode,
    tile_number,
    antenna_number,
):
    if beam_value_mode == "full":
        beam = station.calculate_response(
            frequency=frequency, directions=directions, antenna_mode=antenna_mode
        )
        cbar_title_value = "Full "
    elif beam_value_mode == "tile":
        beam = station.elements[tile_number].calculate_response(
            frequency=frequency,
            directions=directions,
        )
        cbar_title_value = "Tile "
    elif beam_value_mode == "station":
        beam = station.calculate_array_factor(
            frequency=frequency,
            directions=directions,
        )
        cbar_title_value = "Station "
    elif beam_value_mode == "array_factor":
        beam = station.calculate_response(
            frequency=frequency, directions=directions, antenna_mode=None
        )
        cbar_title_value = "Array factor "
    elif beam_value_mode == "element":
        beam = (
            station.elements[tile_number]
            .elements[antenna_number]
            .calculate_response(
                frequency=frequency,
                directions=directions,
                mode=antenna_mode,
            )
        )
        cbar_title_value = "Element "
    else:
        raise ValueError("Not implemented")

    if beam_plot_mode == "power":
        plot_beam = 20 * np.log10(np.abs(beam))
        cbar_title_mode = "power beam (dB)"
    elif beam_plot_mode == "voltage":
        plot_beam = 10 * np.log10(np.abs(beam))
        cbar_title_mode = "voltage beam (dB)"
    elif beam_plot_mode == "real":
        plot_beam = np.real(beam)
        cbar_title_mode = "real part of beam"
    elif beam_plot_mode == "imag":
        plot_beam = np.imag(beam)
        cbar_title_mode = "imaginary part of beam"
    else:
        raise ValueError("Not a permitted beam_plot_mode.")
    return plot_beam, cbar_title_value + cbar_title_mode


def plot_spatial_beam(
    station,
    n_altitude,
    n_azimuth,
    frequency=150e6,
    antenna_mode=None,
    beam_plot_mode="power",
    beam_value_mode="full",
    tile_number=0,
    antenna_number=0,
    points_of_interest=[],
    plot_title=None,
    **kwargs
):
    """
    Helper function to create plots of station beams easily

    Parameters
    ----------
    station : Station object
        Station
    n_altitude : int
        Number of points in the altitude sweep (more is higher resolution)
    n_azimuth : int
        Number of points in the azimuth sweep (similar to n_altitude)
    frequency : float or int, optional
        measurement frequency to plot the beam for, by default 150e6
    antenna_mode : None or str, optional
        Gives the shape of the element beam, by default None
    beam_plot_mode : str, optional
        The way in which the beam should be displayed ("power" of "voltage" in dB and real and imaginary in linear scale), by default "power"
    beam_value_mode : str, optional
        The way in which the beam should be calculated "element", "tile" (array factor), "station" (array factor), "array_factor" (tile and station array factors combined), or "full" (elements and array factors), by default "full"
    tile_number : int, optional
        Which tile in the station to display for tile/element option, by default 0
    antenna_number : int, optional
        Which element in the chosen tile to display (for element option), by default 0
    plot_title: str
        Title of the plot
    points_of_interest : list, optional
        list of directions that should be highlighted in the plot. For example the pointing direction or a bright source, by default [] (no highlighted points)
    """

    # Find the direction unit vectors for the requested sweep ranges
    altitude_sweep = np.linspace(0, np.pi / 2, n_altitude)
    azimuth_sweep = np.linspace(0, 2 * np.pi, n_azimuth)

    AZ, ALT = np.meshgrid(azimuth_sweep, altitude_sweep, indexing="ij")

    directions = np.stack(
        [
            np.cos(ALT).flatten() * np.sin(AZ).flatten(),
            np.cos(ALT).flatten() * np.cos(AZ).flatten(),
            np.sin(ALT).flatten(),
        ],
        axis=0,
    )

    # Calculate the beams
    beam, cbar_title = get_beam(
        beam_value_mode,
        beam_plot_mode,
        station,
        frequency,
        directions,
        antenna_mode,
        tile_number,
        antenna_number,
    )

    ## Create plot
    # Set defaults if these have not been specified
    vmin = kwargs.pop("vmin", -50)
    vmax = kwargs.pop("vmax", 0)
    cmap = kwargs.pop("cmap", "plasma")

    fig, ax = plt.subplots(ncols=1, nrows=1, subplot_kw={"projection": "polar"})
    beam = beam.reshape(AZ.shape)
    im = ax.pcolormesh(
        AZ + np.pi / 2,  # this brings the North up rather than right
        np.cos(ALT),
        beam,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        **kwargs,
    )

    # Add the interesting directions
    for point in points_of_interest:
        # We use inclination rather than altitude, so we get a cosine
        inc = np.arccos(point[2] / np.linalg.norm(point))
        az = np.arctan2(point[0], point[1])
        if np.isnan(
            az
        ):  # Happens in zenith, when East and North are both 0, so we set an arbitrary azimuth
            az = 0
        ax.scatter(az + np.pi / 2, np.sin(inc), color="k", s=100, fc="none")

    if plot_title is not None:
        ax.set_title(plot_title)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
    cbar.set_label(cbar_title)
    ax.set_xlabel(r"South")
    ax.set_ylabel(r"East")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.show()


def plot_spectrotemporal_beam(
    station,
    right_ascension,
    declination,
    frequencies=np.arange(134.1e6, 146.7e6, 195e3),
    utc_starttime="2024-07-04T21:35:00",
    time_resolution=2,
    number_of_timeslots=400,
    antenna_mode=None,
    beam_plot_mode="power",
    beam_value_mode="full",
    tile_number=0,
    antenna_number=0,
    plot_title=None,
    **kwargs
):
    """
    Helper function to create plots of station beams easily

    Parameters
    ----------
    station : Station object
        Station
    right_ascention : float
        Target RA in degrees
    declination : float
        Target dec in degrees
    frequencies : ndarray, optional
        channel central frequencies to plot the beam for in Hz. Default is 195kHz channels between 134.1 and 146.7 MHz (LOFAR-EoR redshift bin 2).
    utc_starttim : str
        Observing time in UTC format (YYYY-MM-DDThh:mm:ss. Default is "2024-07-04T21:35:00"
    time_resolution : float
        Time resolution at which to plot the beam in seconds. Default is 2.
    number_of_timeslots : int
        Number of timeslots (of size time_resolution) to plot. Default is 400
    antenna_mode : None or str, optional
        Gives the shape of the element beam, by default None
    beam_plot_mode : str, optional
        The way in which the beam should be displayed ("power" of "voltage" in dB and real and imaginary in linear scale), by default "power"
    beam_value_mode : str, optional
        The way in which the beam should be calculated "element", "tile" (array factor), "station" (array factor), "array_factor" (tile and station array factors combined), or "full" (elements and array factors), by default "full"
    tile_number : int, optional
        Which tile in the station to display for tile/element option, by default 0
    antenna_number : int, optional
        Which element in the chosen tile to display (for element option), by default 0
    """

    # Find the direction unit vectors for the requested sweep ranges
    directions = [
        station.radec_to_ENU(
            right_ascension=right_ascension,
            declination=declination,
            time=utc_starttime,
            temporal_offset=time_resolution * n,
        )
        for n in range(number_of_timeslots)
    ]
    directions = np.array(directions).T

    beam = np.empty([frequencies.size, number_of_timeslots])
    for channel_number, frequency in enumerate(frequencies):
        # Calculate the beams
        beam[channel_number, :], cbar_title = get_beam(
            beam_value_mode,
            beam_plot_mode,
            station,
            frequency,
            directions,
            antenna_mode,
            tile_number,
            antenna_number,
        )

    T, F = np.meshgrid(
        np.arange(number_of_timeslots) * time_resolution / 60,
        frequencies * 1e-6,
    )

    ## Create plot
    # Set defaults if these have not been specified
    vmin = kwargs.pop("vmin", -50)
    vmax = kwargs.pop("vmax", 0)
    cmap = kwargs.pop("cmap", "plasma")

    fig, ax = plt.subplots(ncols=1, nrows=1)
    im = ax.pcolormesh(
        T,
        F,
        beam,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        **kwargs,
    )

    if plot_title is not None:
        ax.set_title(plot_title)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
    cbar.set_label(cbar_title)
    ax.set_xlabel(r"Time since start of observation (min)")
    ax.set_ylabel(r"Frequency (MHz)")
    fig.show()
