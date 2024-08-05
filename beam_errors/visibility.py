import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import os
from astropy.time import Time, TimeDelta
from astropy import constants as const
from beam_errors.array import calculate_array_factor_contribution
from joblib import Parallel, delayed


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
    prediction_batch_size=10000,
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
    os.makedirs(f"{data_path}/{filename}", exist_ok=True)

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
        prediction_chunk = len(patch.elements) * len(array.keys())
        batch_size = np.max(
            [
                prediction_batch_size // prediction_chunk * prediction_chunk,
                prediction_chunk,  # if the prediction_batch size is set too low, we select 1 prediction_chunk
            ]
        )
        number_of_batches = (len(all_beams) // batch_size) + 1
        first_batch = True

        for batch_number in trange(number_of_batches):
            # Split in time and frequency
            batch = all_beams.iloc[
                batch_number * batch_size : (batch_number + 1) * batch_size
            ]
            if (
                len(batch) == 0
            ):  # in case len(all_beams) is an integer multiple of batch_size, the last batch is empty
                break

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

            # Sum all sources in the patch to get the visibility
            visibility_frame = baseline_frame[
                ["time", "station 1", "station 2", "frequency", "source", "visibility"]
            ]
            visibility = (
                visibility_frame.groupby(
                    ["time", "station 1", "station 2", "frequency"], observed=True
                )
                .agg({"visibility": "sum"})
                .reset_index()
            )

            # Write out the visbility
            visibility.to_csv(
                f"{data_path}/{filename}/{patch_name}.csv",
                index=False,
                header=first_batch,
                mode="w" if first_batch else "a",
            )
            first_batch = False

        os.remove(f"{data_path}/{filename}/temp.csv")
