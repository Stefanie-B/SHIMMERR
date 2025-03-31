import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import pandas as pd
from astropy.time import Time
import os
import pickle


def get_beam(
    beam_value_mode,
    beam_plot_mode,
    station,
    frequency,
    directions,
    phase_center,
    antenna_mode,
    tile_number,
    antenna_number,
    calculate_all_tiles,
):
    """
    Helper function to calculate the beam so it can be plot

    Parameters
    ----------
    beam_value_mode : str, optional
        The way in which the beam should be calculated "element", "tile" (array factor), "station" (array factor), "array_factor" (tile and station array factors combined), or "full" (elements and array factors), by default "full"
    beam_plot_mode : str, optional
        The way in which the beam should be displayed ("power" of "voltage" in dB and real and imaginary in linear scale), by default "power"
    station : Station object
        Station
    frequency : float or int, optional
        measurement frequency to plot the beam for, by default 150e6
    directions : ndarray
        Array with directions in which to calculate the beam, should be 3xNdir
    phase_center : ndarray
        Array with the pointing direction in which to calculate the beam, should be 3xNdir
    antenna_mode : None or str, optional
        Gives the shape of the element beam, by default None
    tile_number : int, optional
        Which tile in the station to display for tile/element option, by default 0
    antenna_number : int, optional
        Which element in the chosen tile to display (for element option), by default 0
    calculate_all_tiles : bool
        If set, all tile beams are computed separately (allows for intra-tile variations), if unset all tile beams are identical (faster computation)

    Returns
    -------
    ndarray
        beam values that can be plotted
    str
        colorbar label
    """
    # Predict the beam in the specified way
    if beam_value_mode == "full":
        beam = station.calculate_response(
            frequency=frequency,
            directions=directions,
            pointing_directions=phase_center,
            antenna_mode=antenna_mode,
            calculate_all_tiles=calculate_all_tiles,
        )
        cbar_title_value = "Full "
    elif beam_value_mode == "tile":
        beam = station.elements[tile_number].calculate_response(
            frequency=frequency, directions=directions, pointing_directions=phase_center
        )
        cbar_title_value = "Tile "
    elif beam_value_mode == "station":
        beam = station.calculate_array_factor(
            frequency=frequency, directions=directions, pointing_directions=phase_center
        )
        cbar_title_value = "Station "
    elif beam_value_mode == "array_factor":
        beam = station.calculate_response(
            frequency=frequency,
            directions=directions,
            antenna_mode=None,
            pointing_directions=phase_center,
            calculate_all_tiles=calculate_all_tiles,
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

    # Parse the beam value
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
    time="2024-01-01T00:00:00",
    calculate_all_tiles=True,
    **kwargs,
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

    phase_center = station.radec_to_ENU(tracking_direction=True, time=time)

    # Calculate the beams
    beam, cbar_title = get_beam(
        beam_value_mode,
        beam_plot_mode,
        station,
        frequency,
        directions,
        phase_center,
        antenna_mode,
        tile_number,
        antenna_number,
        calculate_all_tiles,
    )

    ## Create plot
    # Set defaults if these have not been specified
    vmin = kwargs.pop("vmin", -50)
    vmax = kwargs.pop("vmax", 0)
    cmap = kwargs.pop("cmap", "plasma")
    ax = kwargs.pop("ax", None)
    fig = kwargs.pop("fig", None)

    if ax is None:
        # If no axes provided, create one
        fig, ax = plt.subplots(ncols=1, nrows=1, subplot_kw={"projection": "polar"})
    else:
        # Ensure the axes is in polar projection
        if ax.name != "polar":
            raise ValueError("axes object must be in polar projection")

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
        ax.scatter(az + np.pi / 2, np.sin(inc), color="r", s=100, fc="none")

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
    calculate_all_tiles=True,
    **kwargs,
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
    directions = station.radec_to_ENU(
        right_ascension=right_ascension,
        declination=declination,
        time=utc_starttime,
        temporal_offset=time_resolution,
        number_of_timesteps=number_of_timeslots,
    )

    phase_center = station.radec_to_ENU(
        tracking_direction=True,
        time=utc_starttime,
        temporal_offset=time_resolution,
        number_of_timesteps=number_of_timeslots,
    )

    beam = np.empty([frequencies.size, number_of_timeslots])
    for channel_number, frequency in enumerate(frequencies):
        # Calculate the beams
        beam[channel_number, :], cbar_title = get_beam(
            beam_value_mode,
            beam_plot_mode,
            station,
            frequency,
            directions,
            phase_center,
            antenna_mode,
            tile_number,
            antenna_number,
            calculate_all_tiles,
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

    ax = kwargs.pop("ax", None)
    fig = kwargs.pop("fig", None)

    if ax is None:
        # If no axes provided, create one
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


def plot_visibility(
    file,
    station_pairs,
    magnitude_range=[None, None],
    phase_range=[-np.pi, np.pi],
    **kwargs,
):
    """
    Plots a visibility

    Parameters
    ----------
    file : str
        pth + filename of the visibility
    station_pairs : list
        list of tuples with the baseline to be plotted
    magnitude_range : list, optional
        range of the magnitude scale, by default [None, None]
    phase_range : list, optional
        range of the phase scale, by default [-np.pi, np.pi]
    """
    dataframe = pd.read_csv(file)
    for column in ["time", "frequency", "station 1", "station 2"]:
        dataframe[column] = dataframe[column].astype(pd.CategoricalDtype(ordered=True))

    time_stamps = np.unique(dataframe["time"])
    time_offsets = [Time(time_stamp).unix / 60 for time_stamp in time_stamps]
    time_offsets -= time_offsets[0]

    frequencies = np.unique(dataframe["frequency"]).astype(float)

    F, T = np.meshgrid(frequencies * 1e-6, time_offsets)

    ## Create plot
    fig, axs = plt.subplots(
        ncols=2, nrows=len(station_pairs), figsize=(8, 3 * len(station_pairs))
    )
    for plot_number, station_pair in enumerate(station_pairs):
        visibilities = dataframe.loc[
            (
                (dataframe["station 1"] == station_pair[0])
                & (dataframe["station 2"] == station_pair[1])
            )
            | (
                (dataframe["station 1"] == station_pair[1])
                & (dataframe["station 2"] == station_pair[0])
            ),
            "visibility",
        ]
        visibilities = visibilities.values.astype(complex).reshape(F.shape)

        im = axs[plot_number, 0].pcolormesh(
            T,
            F,
            np.abs(visibilities),
            vmin=magnitude_range[0],
            vmax=magnitude_range[1],
            cmap="viridis",
            **kwargs,
        )

        cbar = fig.colorbar(im, ax=axs[plot_number, 0], orientation="vertical")
        cbar.set_label("|V|")

        im = axs[plot_number, 1].pcolormesh(
            T,
            F,
            np.angle(visibilities),
            vmin=phase_range[0],
            vmax=phase_range[1],
            cmap="hsv",
            **kwargs,
        )

        cbar = fig.colorbar(im, ax=axs[plot_number, 1], orientation="vertical")
        cbar.set_label(r"$\angle$ V")

        axs[plot_number, 0].set_ylabel(f" {'-'.join(station_pair)} \n Frequency (MHz)")
        axs[plot_number, 1].set_yticklabels([])

    for i in range(2):
        axs[-1, i].set_xlabel(r"Time since start of observation (min)")
        for j in range(len(station_pairs) - 1):
            axs[j, i].set_xticklabels([])

    fig.show()


def _plot_convergence_component(results, plot_folder, name, mode):
    """
    Helper function to plot convergence
    """
    plt.figure(figsize=(12, 8))
    for index, (result_label, result_values) in enumerate(results.items()):
        for time_step, result in enumerate(result_values):
            plt.scatter(
                range(result["n_iter"]),
                result[mode],
                color=f"C{index}",
                label=result_label,
                alpha=0.5,
            )
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.semilogy()
    plt.grid()
    os.makedirs(plot_folder, exist_ok=True)
    plt.xlabel("Iteration")
    plt.ylabel(mode + r"(Jy$^2$)")
    if mode == "residuals":
        plt.ylim(top=result[mode][0] * 2)
    plt.savefig(f"{plot_folder}/{mode}_{name}.png")


def plot_convergence(results, plot_folder, name):
    """
    Plot the convergence of a calibration run

    Parameters
    ----------
    results : dict
        Results as output by the DDECal.run (see calibration.py)
    plot_folder : str
        path where the plot should be save
    name : str
        filename of the plot
    """
    for mode in ["loss", "residuals"]:
        _plot_convergence_component(results, plot_folder, name, mode)


def _make_gain_plot(
    gains, direction, frequencies, times, stations, mode, savename, **kwargs
):
    """
    Helper function to plot the gains
    """
    if mode == "amplitude":
        plot_variable = lambda gains, station_number: np.abs(
            gains[:, :, station_number]
        ).T
    elif mode == "phase":
        plot_variable = lambda gains, station_number: np.angle(
            gains[:, :, station_number]
        ).T
    else:
        raise ValueError("Invalid plot mode for gains.")

    ncols = int(np.ceil(np.sqrt(len(stations))))
    if ncols * (ncols - 1) < len(stations):
        nrows = ncols
    else:
        nrows = ncols - 1

    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(15, 9), sharex=True, sharey=True
    )
    ax = np.reshape(ax, (-1))

    times = Time(times)
    times = (times - times[0]).sec
    if len(times) == 1:
        grid_time, grid_freq = np.meshgrid([times[0], times[0] + 1], frequencies / 1e6)
    else:
        grid_time, grid_freq = np.meshgrid(times, frequencies / 1e6)

    # Cycle through stations
    for station_number, station in enumerate(stations):
        # Set correct labels, but hide the labels if they overlap with a different panel
        ax[station_number].set_title(station, y=1, pad=-14)
        ax[station_number].set_xlabel("time (min)")
        ax[station_number].set_ylabel("freq (MHz)")
        ax[station_number].label_outer()

        if len(times) == 1:
            # Plot the desired quantity in heatmap format
            im = ax[station_number].pcolormesh(
                grid_time,
                grid_freq,
                plot_variable(gains, station_number).repeat(2, 1),
                shading="nearest",
                **kwargs,
            )
        else:
            # Plot the desired quantity in heatmap format
            im = ax[station_number].pcolormesh(
                grid_time,
                grid_freq,
                plot_variable(gains, station_number),
                **kwargs,
            )

    # Align panels closely together
    fig.subplots_adjust(wspace=0)
    fig.subplots_adjust(hspace=0)

    fig.suptitle(direction)

    # add colorbar
    fig.colorbar(im, ax=ax.ravel().tolist())

    # Save
    fig.savefig(savename)
    plt.close("all")


def _make_gains_gif(gain_folder, metadata, mode):
    """
    Helper function to create a gif of the gain plots that cycles through directions
    """
    fig = plt.figure(figsize=(15, 9), dpi=300)
    plt.axis("off")
    fig.tight_layout()

    img_list = []
    for direction in metadata["directions"]:
        fname = f"{gain_folder}/{direction}_{mode}.png"
        file = plt.imread(fname)
        img = plt.imshow(file)
        img_list.append([img])
    video_name = f"{gain_folder}/{mode}.gif"

    ani = animation.ArtistAnimation(fig, img_list, blit=True, repeat_delay=1000)
    ani.save(video_name, fps=1, dpi=100)


def plot_gains(
    fname, plot_folder, name, amplitude_lims=[0, 2], phase_lims=[-np.pi, np.pi]
):
    """
    Function to plot gains

    Parameters
    ----------
    fname : str
        path + filename of the results object that contians the gails
    plot_folder : str
        Folder where the plot should be saved
    name : str
        name of the plot
    amplitude_lims : list, optional
        range of the gain amplitude in the plot, by default [0, 2]
    phase_lims : list, optional
        range of the gain phase in the plot, by default [-np.pi, np.pi]
    """
    with open(fname, "rb") as fp:
        full_results = pickle.load(fp)
    with open(f"{fname}_metadata", "rb") as fp:
        metadata = pickle.load(fp)

    gains = np.array(
        [result["gains"] for result in full_results]
    )  # time, freq_sols, stations, dirs

    os.makedirs(f"{plot_folder}/{name}", exist_ok=True)
    for plot_number, direction in enumerate(metadata["directions"]):
        _make_gain_plot(
            gains[:, :, :, plot_number],
            direction,
            metadata["frequencies"],
            metadata["times"],
            metadata["stations"],
            mode="amplitude",
            savename=f"{plot_folder}/{name}/{direction}_amplitude.png",
            vmin=amplitude_lims[0],
            vmax=amplitude_lims[1],
            cmap="viridis",
        )
        _make_gain_plot(
            gains[:, :, :, plot_number],
            direction,
            metadata["frequencies"],
            metadata["times"],
            metadata["stations"],
            mode="phase",
            savename=f"{plot_folder}/{name}/{direction}_phase.png",
            vmin=phase_lims[0],
            vmax=phase_lims[1],
            cmap="hsv",
        )
    _make_gains_gif(f"{plot_folder}/{name}", metadata, "amplitude")
    _make_gains_gif(f"{plot_folder}/{name}", metadata, "phase")


def _load_gains(fname_gains):
    """
    Helper function to read gains from disk
    """
    with open(fname_gains, "rb") as fp:
        full_results = pickle.load(fp)
    gains = np.array(
        [result["gains"] for result in full_results]
    )  # time, freq_sols, stations, dirs
    return gains


def _parse_true_gains(gains, reference_station, metadata):
    """
    Helper function to read true gains (ratio between true beam response and calibration beam response) and remove unitairy ambiguity, such that this can mach estimated gains.
    """
    if reference_station is not None:
        amplitudes = np.abs(gains)
        phases = np.angle(gains)

        reference_phases = np.squeeze(
            phases[:, :, metadata["stations"] == reference_station, :]
        )
        phases -= reference_phases[:, :, np.newaxis, :]
        true_gains = amplitudes * np.exp(1j * phases)
    return true_gains


def _parse_gain_error(true_gains, estimated_gains):
    """
    Helper function to calculate the error in estimated gains from the true gain solution
    """
    temporal_resolution_factor = true_gains.shape[0] // estimated_gains.shape[0]
    spectral_resolution_factor = true_gains.shape[1] // estimated_gains.shape[1]

    expanded_gains = estimated_gains.repeat(temporal_resolution_factor, axis=0).repeat(
        spectral_resolution_factor, axis=1
    )
    plot_gains = expanded_gains / true_gains
    return plot_gains


def plot_gain_error(
    fname_gains,
    fname_true_gains,
    plot_folder,
    name,
    reference_station="CS002HBA0",
    amplitude_lims=[0, 2],
    phase_lims=[-np.pi, np.pi],
):
    """
    Creates a plot of the error in estimated gains using the ideal gains (ratio between true beam response and claibration beam)

    Parameters
    ----------
    fname_gains : str
        path + filename where the estimated gains are stored
    fname_true_gains : str
        path + filename where the ideal gains are stored (see metrics.py)
    plot_folder : str
        directory where the plot is to be stored
    name : str
        filename of the plot
    reference_station : str, optional
        name fo the reference station used to remove unitairy ambiguity by using it as zero-phase, by default "CS002HBA0"
    amplitude_lims : list, optional
        range of amplitude colorbar, by default [0, 2]
    phase_lims : list, optional
        range of phase colorbar, by default [-np.pi, np.pi]
    """
    with open(f"{fname_true_gains}_metadata", "rb") as fp:
        metadata = pickle.load(fp)

    estimated_gains = _load_gains(fname_gains)
    true_gains = _parse_true_gains(
        _load_gains(fname_true_gains), reference_station, metadata
    )

    plot_gains = _parse_gain_error(true_gains, estimated_gains)

    os.makedirs(f"{plot_folder}/{name}", exist_ok=True)
    for plot_number, direction in enumerate(metadata["directions"]):
        _make_gain_plot(
            plot_gains[:, :, :, plot_number],
            direction,
            metadata["frequencies"],
            metadata["times"],
            metadata["stations"],
            mode="amplitude",
            savename=f"{plot_folder}/{name}/{direction}_amplitude.png",
            vmin=amplitude_lims[0],
            vmax=amplitude_lims[1],
            cmap="bwr",
        )
        _make_gain_plot(
            plot_gains[:, :, :, plot_number],
            direction,
            metadata["frequencies"],
            metadata["times"],
            metadata["stations"],
            mode="phase",
            savename=f"{plot_folder}/{name}/{direction}_phase.png",
            vmin=phase_lims[0],
            vmax=phase_lims[1],
            cmap="hsv",
        )
    _make_gains_gif(f"{plot_folder}/{name}", metadata, "amplitude")
    _make_gains_gif(f"{plot_folder}/{name}", metadata, "phase")
