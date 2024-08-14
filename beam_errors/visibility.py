import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import os
from astropy.time import Time, TimeDelta
from astropy import constants as const
from beam_errors.array import calculate_array_factor_contribution
from joblib import Parallel, delayed
import shutil


def get_time_labels(start_time_utc, duration, time_resolution):
    """
    Helper function to generate time labels for the visibilities
    """
    number_of_timeslots = int(duration * 3600 // time_resolution)
    time_of_observation = Time(start_time_utc)
    time_offset = TimeDelta(time_resolution, format="sec")
    time_labels = [
        (time_of_observation + n * time_offset).isot for n in range(number_of_timeslots)
    ]
    return time_labels


def calculate_directions(
    station,
    start_time_utc,
    time_resolution,
    number_of_timeslots,
    right_ascensions,
    declinations,
):
    """
    Helper function to generate source directions and tracking directions in an easy format
    """
    tracking_directions = station.radec_to_ENU(
        time=start_time_utc,
        temporal_offset=time_resolution,
        number_of_timesteps=number_of_timeslots,
        tracking_direction=True,
    )
    directions = Parallel(n_jobs=-1)(
        delayed(station.radec_to_ENU)(
            right_ascension=right_ascension,
            declination=declination,
            time=start_time_utc,
            temporal_offset=time_resolution,
            number_of_timesteps=number_of_timeslots,
        )
        for right_ascension, declination in zip(right_ascensions, declinations)
    )
    return np.concatenate(directions, axis=1), np.tile(
        tracking_directions, len(right_ascensions)
    )


def record_beams(
    df, new_batch, first_frequency, last_frequency, first_station, file_path
):
    """
    Helper function to write the temporary beam files
    """
    # Initialize the station with the first frequency, otherwise append new data
    if first_frequency:
        df = new_batch.copy()
    else:
        df = pd.concat([df, new_batch])

    # When all frequencies are done, write the temporary file
    if last_frequency:
        if first_station:
            df.to_csv(f"{file_path}/temp.csv", index=False)
        else:
            df.to_csv(
                f"{file_path}/temp.csv",
                index=False,
                header=False,
                mode="a",
            )

    return df


def process_batch(batch, source_powers):
    # Create a dataframe to do baseline computations on (such that all combinations of stations are present at each interval)
    baseline_frame = batch.merge(
        batch, on=["time", "frequency", "source"], suffixes=(" 1", " 2")
    )

    # Remove duplicate baselines (if i-j is included, j-i is removed)
    baseline_frame = baseline_frame[
        baseline_frame["station 1"] <= baseline_frame["station 2"]
    ]

    # Complex array response towards each source (value of station 1 times conj(value) of station 2)
    baseline_frame["visibility"] = baseline_frame["value 1"] * np.conj(
        baseline_frame["value 2"]
    )
    baseline_frame.loc[:, "visibility"] = baseline_frame[
        "visibility"
    ] * baseline_frame.apply(
        lambda row: source_powers[(row["source"], row["frequency"])], axis=1
    )
    baseline_frame["baseline"] = (
        baseline_frame["station 1"].astype(str)
        + "-"
        + baseline_frame["station 2"].astype(str)
    )

    # Sum all sources in the patch to get the visibility
    visibility_frame = baseline_frame[
        ["time", "baseline", "frequency", "source", "visibility"]
    ]
    visibility = (
        visibility_frame.groupby(["time", "frequency", "baseline"], observed=True)
        .agg({"visibility": "sum"})
        .reset_index()
    )
    return visibility


def predict_patch_visibilities(
    array,
    skymodel,
    frequencies,
    start_time_utc,
    filename,
    data_path="./data/predictions",
    time_resolution=2,
    duration=12,
    antenna_mode="omnidirectional",
    basestation=None,
    reuse_tile_beam=False,
):
    """
    Predicts noiseless visibilities of each patch and saves each patch to disk.

    Parameters
    ----------
    array : dict
        Dictionary containing Station objects
    skymodel : Skymodel object
        skymodel to be simulated
    frequencies : list or ndarray of floats
        Frequencies at which to simulate the array. We simulate no smearing etc.
    start_time_utc : str
        UTC start time of the observation in format YYYY-MM-DDThh:mm:ss, example: 2024-07-04T19:25:00
    filename : str
        name of the folder in which the visibilities should be saved
    data_path : str, optional
        folder in which the destination folder should be saved, by default "./data/predictions"
    time_resolution : int, optional
        time resolution in seconds, by default 2
    duration : int, optional
        observation duration in hours, by default 12
    antenna_mode : str, optional
        beam mode for the lowest level elements, by default "omnidirectional"
    basestation : str, optional
        name of the station used as the 'array center', by default None (selects the first station in the array dictionary)
    prediction_batch_size : int, optional
        approximate batch size for each visibility prediction after beam calculation. This is a trade off between speed and memory virtualization. The batch will become an integer multiple of the number of stations * number of sources, so this number may be exceeded if one of such batches is larger than the specified value, by default 10000.
    """
    os.makedirs(f"{data_path}/{filename}/patch_models", exist_ok=True)

    # Create time labels to write for the data later
    time_labels = get_time_labels(start_time_utc, duration, time_resolution)
    number_of_timeslots = int(duration * 3600 // time_resolution)

    # We set station positions in the ENU system of the base station
    if basestation is None:
        basestation = list(array.keys())[0]
    [station.set_array_position(array[basestation]) for station in array.values()]

    for patch_name, patch in skymodel.elements.items():
        print(f"Simulating patch {patch_name}. Calculating station beams...")

        right_ascensions = [source.ra for source in patch.elements.values()]
        declinations = [source.dec for source in patch.elements.values()]

        # For each patch, get the directions of all sources in the basestation's ENU system
        basestation_directions, basestation_phase_center = calculate_directions(
            array[basestation],
            start_time_utc,
            time_resolution,
            number_of_timeslots,
            right_ascensions,
            declinations,
        )
        basestation_directions -= basestation_phase_center

        all_beams = None
        for station_name, station in tqdm(array.items()):
            # Directions (offset from pointing) in station's ENU system
            directions, phase_center = calculate_directions(
                station,
                start_time_utc,
                time_resolution,
                number_of_timeslots,
                right_ascensions,
                declinations,
            )

            for frequency in frequencies:
                # Beam of the station itself
                station_beam = station.calculate_response(
                    frequency=frequency,
                    directions=directions,
                    pointing_directions=phase_center,
                    antenna_mode=antenna_mode,
                    calculate_all_tiles=not reuse_tile_beam,
                )

                # Geometric delay for the station
                k = 2 * np.pi * frequency / const.c.value
                delay = np.squeeze(
                    calculate_array_factor_contribution(
                        station.p_array.reshape(1, 3),
                        np.array([station.g]),
                        k,
                        basestation_directions,
                    )
                )

                # Set sources below the horizon to 0
                station_beam[np.isnan(station_beam)] = 0

                # Record all responses
                df = pd.DataFrame(
                    {
                        "station": pd.Categorical(
                            np.repeat(station_name, np.size(station_beam))
                        ),
                        "time": pd.Categorical(
                            np.tile(time_labels, len(patch.elements))
                        ),
                        "source": pd.Categorical(
                            np.repeat(list(patch.elements.keys()), number_of_timeslots)
                        ),
                        "frequency": pd.Categorical(
                            np.repeat(frequency, np.size(station_beam))
                        ),
                        "value": station_beam * delay,
                    },
                )

                # Writing each station separately reduces memory virtualization and therefore speeds up the code
                # despite the time spent on writing to csv. For very large datasets, earlier writing may be needed
                all_beams = record_beams(
                    all_beams,
                    df,
                    frequency == frequencies[0],
                    frequency == frequencies[-1],
                    station_name == list(array.keys())[0],
                    f"{data_path}/{filename}",
                )

        print("Calculating visibilities...")

        # Load all beams and group them such that we can split in time and frequency
        all_beams = pd.read_csv(f"{data_path}/{filename}/temp.csv")
        all_beams["value"] = all_beams["value"].astype(complex)
        for column in ["time", "frequency", "station", "source"]:
            all_beams[column] = all_beams[column].astype(
                pd.CategoricalDtype(ordered=True)
            )
        all_beams = all_beams.sort_values(by=["time", "frequency"])

        # Load the sources and precompute the powers at the measured frequencies
        source_powers = {
            (source, freq): patch.elements[source].I(freq)
            for source in all_beams["source"].unique()
            for freq in all_beams["frequency"].unique()
        }

        # Per processing step, we need to have all sources in a patch and all stations, so the chunk must be an integer multiple of this (at at least 1 batch)
        batch_size = len(patch.elements) * len(array.keys())
        batches = [
            all_beams.iloc[i * batch_size : (i + 1) * batch_size]
            for i in range(len(all_beams) // batch_size)
        ]

        visibilities = Parallel(n_jobs=-1)(
            delayed(process_batch)(batch, source_powers) for batch in tqdm(batches)
        )

        for i, df in enumerate(visibilities):
            df.to_csv(
                f"{data_path}/{filename}/patch_models/{patch_name}.csv",
                index=False,
                header=(i == 0),
                mode="w" if i == 0 else "a",
            )

        os.remove(f"{data_path}/{filename}/temp.csv")


def add_thermal_noise(filename, data_path, sefd):
    file_name_in = f"{data_path}/{filename}/full_model.csv"
    file_name_out = f"{data_path}/{filename}/data.csv"
    first_batch = True
    rng = np.random.default_rng()
    for batch in pd.read_csv(file_name_in, chunksize=int(1e5)):
        if first_batch:
            d_frequency = np.diff(np.unique(batch["frequency"]).astype(float))[0]
            times = np.unique(batch["time"])
            t1 = Time(times[0])
            t2 = Time(times[1])
            d_time = (t2 - t1).sec
            std_dev = sefd / np.sqrt(2 * d_time * d_frequency)
        visibility = batch["visibility"].values.astype(complex)
        noise = rng.standard_normal(2 * visibility.size) * std_dev
        complex_noise = noise[: visibility.size] + 1j * noise[visibility.size :]
        noisy_visibility = visibility + complex_noise.reshape(visibility.shape)

        batch["visibility"] = noisy_visibility
        batch.to_csv(
            file_name_out,
            index=False,
            header=first_batch,
            mode="w" if first_batch else "a",
        )
        first_batch = False


def sum_patches(filename, data_path, skymodel):

    model_directory = f"{data_path}/{filename}/patch_models/"
    file_name_out = f"{data_path}/{filename}/full_model.csv"

    patch_names = list(skymodel.elements.keys())
    all_files = [f"{model_directory}{patch_name}.csv" for patch_name in patch_names]
    patch_dataframes = [pd.read_csv(file) for file in all_files]
    for patch_dataframe in patch_dataframes:
        patch_dataframe["visibility"] = patch_dataframe["visibility"].astype(complex)

    full_model = (
        pd.concat(patch_dataframes)
        .groupby(["time", "frequency", "baseline"], observed=True)
        .agg({"visibility": "sum"})
        .reset_index()
    )

    full_model.to_csv(
        file_name_out,
        index=False,
        header=True,
        mode="w",
    )


def predict_data(
    array,
    skymodel,
    frequencies,
    start_time_utc,
    filename,
    data_path="./data/predictions",
    time_resolution=2,
    duration=12,
    antenna_mode="omnidirectional",
    basestation=None,
    reuse_tile_beam=False,
    SEFD=4.2e3,
):
    predict_patch_visibilities(
        array=array,
        skymodel=skymodel,
        frequencies=frequencies,
        start_time_utc=start_time_utc,
        filename=filename,
        data_path=data_path,
        time_resolution=time_resolution,
        duration=duration,
        antenna_mode=antenna_mode,
        basestation=basestation,
        reuse_tile_beam=reuse_tile_beam,
    )

    sum_patches(filename, data_path, skymodel)

    if SEFD is not None:
        add_thermal_noise(filename, data_path, SEFD)
    else:
        shutil.copyfile(
            f"{data_path}/{filename}/full_model.csv", f"{data_path}/{filename}/data.csv"
        )
