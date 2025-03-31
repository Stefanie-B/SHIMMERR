import casacore.tables as tab
import os
from pathlib import Path
import numpy as np
from astropy.time import Time
import shutil
import csv


def _create_output_file(array, template_name, output_file, DP3_binding):
    """
    Creates an empty MS to store the visibilities in
    """
    # Copy the template to a temporary file that can be edited and remove unused stations
    shutil.rmtree(output_file, ignore_errors=True)
    template = tab.table(template_name, readonly=True, memorytable=True, ack=False)
    out_ms = template.copy(f"{output_file}_temp", deep=True, valuecopy=True)
    antenna_names_ms = out_ms.ANTENNA.NAME[:]
    template.close()
    out_ms.close()

    # Make list of elements that must be deleted
    antenna_names_array = array.keys()
    removable_elements = [
        antenna for antenna in antenna_names_ms if antenna not in antenna_names_array
    ]

    if len(removable_elements) > 0:
        # Write the MS with the correct stations
        filtered_baselines = f"^{','.join(removable_elements)}&&*"
        os.system(
            f"{DP3_binding} msin={output_file}_temp msout={output_file} steps=[filter] verbosity=quiet showprogress=False filter.remove=1 filter.type=filter filter.baseline='{filtered_baselines}'"
        )
    else:
        shutil.copytree(f"{output_file}_temp", output_file)
    shutil.rmtree(f"{output_file}_temp")


def _unflag_broken_tiles(array, output_file):
    """
    Resets the flags in the MS such that all remote stations use the 24 inner tiles, the core stations are split, and no other flags are present
    """
    antenna_table = tab.table(
        f"{output_file}::LOFAR_ANTENNA_FIELD", readonly=False, ack=False
    )
    for station_number in range(len(array)):
        station_type = antenna_table.NAME.getcell(station_number)
        if station_type == "HBA0":
            flag_indices = range(24)
        elif station_type == "HBA1":
            flag_indices = range(24, 48)
        else:
            flag_indices = [
                5,
                6,
                10,
                11,
                12,
                13,
                17,
                18,
                19,
                20,
                21,
                22,
                25,
                26,
                27,
                28,
                29,
                30,
                34,
                35,
                36,
                37,
                41,
                42,
            ]
        element_flags = np.ones([48, 2], dtype=bool)
        element_flags[flag_indices, :] = False
        antenna_table.putcell("ELEMENT_FLAG", station_number, element_flags)
    antenna_table.close()

    # Also unflag all cells
    out_ms = tab.table(output_file, readonly=False, ack=False)
    tab.taql("UPDATE $out_ms SET FLAG=F, FLAG_ROW=F, WEIGHT_SPECTRUM=1")
    out_ms.close()


def _reshape_columns(table_name, column_names, new_shape):
    """
    Helper function to add empty rows (for example, for added timesteps or channels) to the observation.
    """
    table = tab.table(table_name, readonly=False, ack=False)
    for column_name in column_names:
        desc = table.getcoldesc(column_name)
        table.removecols(column_name)
        desc["shape"] = new_shape
        col_desc = tab.makecoldesc(column_name, desc)
        table.addcols(col_desc)
    table.close()


def _create_frequency_channels(output_file, frequencies):
    """
    Sets the channels correctly in the MS
    """
    subband_name = "CUSTOM_SB"
    n_freqs = len(frequencies)
    d_freq = abs(frequencies[1] - frequencies[0])
    chan_width = np.repeat(d_freq, n_freqs)
    total_bandwidth = n_freqs * d_freq

    # First fix the spectral_window table
    # Some columns must be modified to accomodate a different number of channels
    _reshape_columns(
        table_name=f"{output_file}::SPECTRAL_WINDOW",
        column_names=["CHAN_FREQ", "CHAN_WIDTH", "EFFECTIVE_BW", "RESOLUTION"],
        new_shape=(n_freqs,),
    )

    # Now also reshape the data fields
    _reshape_columns(
        table_name=f"{output_file}",
        column_names=["DATA", "FLAG", "WEIGHT_SPECTRUM"],
        new_shape=(n_freqs, 4),
    )

    out_ms = tab.table(output_file, readonly=False, ack=False)
    tab.taql(
        """UPDATE $out_ms::SPECTRAL_WINDOW SET NAME=$subband_name,
        NUM_CHAN=$n_freqs, TOTAL_BANDWIDTH=$total_bandwidth,
        CHAN_FREQ=$frequencies, CHAN_WIDTH=$chan_width,
        EFFECTIVE_BW=$chan_width,RESOLUTION=$chan_width"""
    )
    tab.taql("UPDATE $out_ms SET FLAG=F, FLAG_ROW=F, WEIGHT_SPECTRUM=1")
    out_ms.close()


def _expand_in_time(output_file, times):
    """
    Expands the MS in time
    """
    # metadata
    t_mjd = [Time(time).mjd * 24 * 3600 for time in times]
    start_time = t_mjd[0]
    end_time = t_mjd[-1]
    time_centroid = (t_mjd[-1] - t_mjd[0]) / 2
    time_resolution = t_mjd[1] - t_mjd[0]
    n_times = len(times)

    out_ms = tab.table(output_file, readonly=False, ack=False)
    tab.taql(
        f"""UPDATE $out_ms SET TIME=$start_time, TIME_CENTROID=$start_time,
              EXPOSURE={time_resolution}, INTERVAL={time_resolution}"""
    )
    tab.taql("UPDATE $out_ms::FEED SET TIME=$time_centroid")
    tab.taql("UPDATE $out_ms::FIELD SET TIME=$time_centroid")
    tab.taql(
        """UPDATE $out_ms::OBSERVATION SET LOFAR_OBSERVATION_START=$start_time,
                LOFAR_OBSERVATION_END=$end_time"""
    )

    N_bl = len(out_ms)

    # Make a copy of the single timestep so we can reinsert that at the different time stels
    template = out_ms.copy(f"{output_file}_temp", deep=True, valuecopy=True)
    for t, time in enumerate(t_mjd[1:]):
        tab.taql("INSERT INTO $out_ms SELECT FROM $template")
        offset = (t + 1) * N_bl
        tab.taql("UPDATE $out_ms SET TIME=$time, TIME_CENTROID=$time OFFSET $offset")
    out_ms.close()
    template.close()
    if os.path.islink(f"{output_file}_temp"):
        os.unlink(f"{output_file}_temp")
    elif os.path.isdir(f"{output_file}_temp"):
        shutil.rmtree(f"{output_file}_temp")


def _adjust_pointing(array, output_file):
    """
    Sets the pointing of the interferometer in the MS
    """
    dec, ra = list(array.values())[0].d.values()
    pointing = np.array([[ra, dec]]) / 180 * np.pi

    field_table = tab.table(f"{output_file}::FIELD", readonly=False, ack=False)
    field_table.putcol("DELAY_DIR", pointing[np.newaxis])
    field_table.putcol("PHASE_DIR", pointing[np.newaxis])
    field_table.putcol("REFERENCE_DIR", pointing[np.newaxis])
    field_table.putcol("LOFAR_TILE_BEAM_DIR", pointing[np.newaxis])
    field_table.close()


def _get_ms_order(output_file, baselines):
    """
    Helper function to copy the order of the antennas from the MS to the data, such that the data can be ordered in the same way.
    """
    out_ms = tab.table(output_file, readonly=False, ack=False)
    antenna_names_ms = out_ms.ANTENNA.NAME[:]
    ms_order = [
        np.argwhere(
            (antenna_names_ms[row["ANTENNA1"]] == baselines[:, 0])
            * (antenna_names_ms[row["ANTENNA2"]] == baselines[:, 1])
        )[0, 0]
        for row in out_ms[: baselines.size // 2]
    ]
    out_ms.close()
    return ms_order


def _export_visibilities(output_file, visibilities, n_freqs, ms_order):
    """
    Exports the data to the MS
    """
    # Reorder the baselines to MS standard
    reordered_visibilities = visibilities[:, :, ms_order]

    # Create a stack of visibilities for Stokes I = (XX+YY)/2, so the data goes in as [data, 0, 0, data]
    stacked_visibilities = reordered_visibilities.transpose(0, 2, 1).reshape(
        -1, n_freqs
    )
    Stokes_visibilities = np.stack(
        [
            stacked_visibilities,
            np.zeros_like(stacked_visibilities),
            np.zeros_like(stacked_visibilities),
            stacked_visibilities,
        ],
        axis=2,
    )
    out_ms = tab.table(output_file, readonly=False, ack=False)
    out_ms.putcol("DATA", Stokes_visibilities)
    out_ms.close()
    return ms_order


def _export_uvw_coordinates(output_file, array, times, baselines, ms_order):
    """
    Sets the station UVW coordinates (station positions) in the MS
    """
    reordered_baselines = baselines[ms_order, :]

    # Obtain the Earth rotation angle to find the rotation of the UV-plane
    n_times = len(times)
    time_objects = [Time(time) for time in times]
    julian_dates = np.array([t.mjd - 51544.5 for t in time_objects])
    ERA = 2 * np.pi * (0.7790572732640 + 1.00273781191135448 * julian_dates)

    # Calculate the unit vectors of the uv-plane at each time
    station = list(array.values())[0]
    declination = np.tile(station.d["dec"] / 180 * np.pi, n_times)
    u_unit = np.array([np.sin(ERA), np.cos(ERA), np.zeros_like(ERA)])
    v_unit = np.array(
        [
            -np.sin(declination) * np.cos(ERA),
            np.sin(declination) * np.sin(ERA),
            np.cos(declination),
        ]
    )
    w_unit = np.array(
        [
            -np.cos(declination) * np.cos(ERA),
            np.cos(declination) * np.sin(ERA),
            -np.sin(declination),
        ]
    )

    # obtain uv-coordinates by projecting onto UVW-unit vectors
    n_baselines = reordered_baselines.size // 2
    uvw = np.empty([n_times * n_baselines, 3], dtype=float)
    for baseline_number, (station_1, station_2) in enumerate(reordered_baselines):
        baseline_vector = array[station_2].p - array[station_1].p

        uvw[baseline_number::n_baselines, 0] = -np.dot(baseline_vector, u_unit)
        uvw[baseline_number::n_baselines, 1] = -np.dot(baseline_vector, v_unit)
        uvw[baseline_number::n_baselines, 2] = np.dot(baseline_vector, w_unit)

    # Place the column
    out_ms = tab.table(output_file, readonly=False, ack=False)
    out_ms.putcol("UVW", uvw)
    out_ms.close()


def export_MS(
    array,
    visibility_file,
    output_file,
    template_name=f"{Path(__file__).parent.parent}/files/LOFAR_HBA_Dutch_template.MS",
    DP3_binding="DP3",
):
    """
    Function that converts visbility CSVs from SHIMMERR into MS.

    Parameters
    ----------
    array : dict
        dictionary of Station objects that form the interferometer
    visibility_file : str
        path + filename of the csv with the stored visibilities
    output_file : str
        path + filename of the MS in which the visibilities should be output
    template_name : str, optional
        path + filename of the MS template. Currently SHIMMERR is only packages with LOFAR HBA, but another template van be supplied by the user. This will change the ELEMENT_FLAGGING, however, (see the _unflag_broken_tiles function), by default f"<SHIMMERR home directory>/files/LOFAR_HBA_Dutch_template.MS"
    DP3_binding : str, optional
        binding to call DP3 from (for example if it should be called with a container), by default "DP3"
    """
    # Load the data so the metadata is known
    with open(visibility_file) as csv_file:
        data = list(csv.DictReader(csv_file))

        frequencies = np.unique([float(row["frequency"]) for row in data])
        times = np.unique([row["time"] for row in data])
        visibilities = [complex(row["visibility"]) for row in data]
        baselines = np.unique(
            [tuple(row["baseline"].split("-")) for row in data], axis=0
        )
    visibilities = np.array(visibilities).reshape(len(times), len(frequencies), -1)

    # create the ms
    _create_output_file(array, template_name, output_file, DP3_binding)
    _unflag_broken_tiles(array, output_file)
    _create_frequency_channels(output_file, frequencies)
    _expand_in_time(output_file, times)
    _adjust_pointing(array, output_file)
    ms_order = _get_ms_order(output_file, baselines)
    _export_visibilities(output_file, visibilities, len(frequencies), ms_order)
    _export_uvw_coordinates(output_file, array, times, baselines, ms_order)
