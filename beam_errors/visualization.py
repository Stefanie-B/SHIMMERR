import numpy as np
from numba import jit, complex128, float64
from matplotlib import pyplot as plt

c = 299792458  # m/s


@jit(
    complex128[:, :](
        float64[:, :],
        complex128[:],
        float64,
        float64[:],
        float64[:, :],
    ),
    fastmath=True,
    nopython=True,
)
def calculate_array_factor_contribution(positions, gains, k, pointing, directions):
    positions = np.ascontiguousarray(positions)
    directions = np.ascontiguousarray(directions)
    gains = np.ascontiguousarray(gains)
    pointing = np.ascontiguousarray(pointing)
    relative_pointing = directions - pointing[:, np.newaxis]
    phase_delay = k * (positions @ relative_pointing)
    return gains[:, np.newaxis] * np.exp(1j * phase_delay)


def calculate_element_beam(element, frequency, directions, antenna_mode):
    antenna_beam = [
        element.calculate_response(
            direction=directions[:, i], frequency=frequency, mode=antenna_mode
        )
        for i in range(directions.shape[1])
    ]

    antenna_beam = np.array(antenna_beam)
    return antenna_beam


def calculate_tile_beam(tile, frequency, directions):
    k = 2 * np.pi * frequency / c

    antenna_factors = calculate_array_factor_contribution(
        positions=tile.get_element_property("p_ENU"),
        gains=tile.get_element_property("g"),
        k=k,
        pointing=tile.d,
        directions=directions,
    )

    tile_beam = np.mean(antenna_factors, axis=0)

    return tile_beam


def calculate_station_beam(station, frequency, directions, tile_beams=None):
    k = 2 * np.pi * frequency / c

    tile_factors = calculate_array_factor_contribution(
        positions=station.get_element_property("p_ENU"),
        gains=station.get_element_property("g"),
        k=k,
        pointing=station.d,
        directions=directions,
    )

    if tile_beams is None:
        station_beam = np.mean(tile_factors, axis=0)
    else:
        tile_beams = np.array(tile_beams)
        tile_responses = tile_beams * tile_factors
        station_beam = np.mean(tile_responses, axis=0)

    return station_beam


def create_beam_plot(
    azimuths,
    altitudes,
    beam,
    title=None,
    cbar_title=None,
    points_of_interest=[],
    **kwargs
):
    vmin = kwargs.pop("vmin", -50)
    vmax = kwargs.pop("vmax", 0)
    cmap = kwargs.pop("cmap", "plasma")

    fig, ax = plt.subplots(ncols=1, nrows=1, subplot_kw={"projection": "polar"})

    beam = beam.reshape(azimuths.shape)

    im = ax.pcolormesh(
        azimuths + np.pi / 2,  # this brings the North up rather than right
        np.cos(altitudes),
        beam,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        **kwargs,
    )

    for point in points_of_interest:
        inc = np.arccos(point[2] / np.linalg.norm(point))
        az = np.arctan2(point[0], point[1])
        if np.isnan(az):
            az = 0
        ax.scatter(az + np.pi / 2, np.sin(inc), color="k", s=100, fc="none")

    if title is not None:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical")
    cbar.set_label(cbar_title)
    ax.set_xlabel(r"South")
    ax.set_ylabel(r"East")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.show()


def plot_beam(
    station,
    n_altitude,
    n_azimuth,
    frequency=150e6,
    antenna_mode=None,
    beam_plot_mode="power",
    beam_value_mode="full",
    tile_number=0,
    antenna_number=0,
    **kwargs
):
    # We calculate all directions for a single element instead of the other way around
    # because this function uses a lot of directions (more than the expected number of elements)
    # Set all east, north, up coordinates
    station.set_ENU_positions()

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

    if beam_value_mode == "full":
        tile_beams = [
            calculate_tile_beam(tile=tile, frequency=frequency, directions=directions)
            for tile in station.elements
        ]
        station_beam = calculate_station_beam(
            station=station,
            frequency=frequency,
            directions=directions,
            tile_beams=tile_beams,
        )
        antenna_beam = calculate_element_beam(
            element=station.elements[tile_number].elements[antenna_number],
            frequency=frequency,
            directions=directions,
            antenna_mode=antenna_mode,
        )
        beam = station_beam * antenna_beam
        cbar_title_value = "Full "
    elif beam_value_mode == "tile":
        beam = calculate_tile_beam(
            tile=station.elements[tile_number],
            frequency=frequency,
            directions=directions,
        )
        cbar_title_value = "Tile "
    elif beam_value_mode == "station":
        beam = calculate_station_beam(
            station=station,
            frequency=frequency,
            directions=directions,
        )
        cbar_title_value = "Station "
    elif beam_value_mode == "array_factor":
        tile_beams = [
            calculate_tile_beam(tile=tile, frequency=frequency, directions=directions)
            for tile in station.elements
        ]
        beam = calculate_station_beam(
            station=station,
            frequency=frequency,
            directions=directions,
            tile_beams=tile_beams,
        )
        cbar_title_value = "Array factor "
    elif beam_value_mode == "element":
        beam = calculate_element_beam(
            element=station.elements[tile_number].elements[antenna_number],
            frequency=frequency,
            directions=directions,
            antenna_mode=antenna_mode,
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

    create_beam_plot(
        azimuths=AZ,
        altitudes=ALT,
        beam=plot_beam,
        cbar_title=cbar_title_value + cbar_title_mode,
        points_of_interest=[station.d],
        **kwargs,
    )
