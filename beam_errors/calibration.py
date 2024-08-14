import copy
from beam_errors.visibility import predict_patch_visibilities
import csv
import numpy as np
from astropy import constants as const
from astropy.time import Time
from scipy.linalg import solve_triangular
from joblib import Parallel, delayed
import os


def predict_model(
    array,
    skymodel,
    frequencies,
    start_time_utc,
    visibility_file,
    time_resolution,
    duration,
    antenna_mode,
    basestation,
):
    unit_gain_array = copy.deepcopy(array)
    for station in unit_gain_array.values():
        station.reset_elements()
        station.g = 1.0 + 0j

    data_path = visibility_file.rstrip("/data.csv")

    predict_patch_visibilities(
        array=unit_gain_array,
        skymodel=skymodel,
        frequencies=frequencies,
        start_time_utc=start_time_utc,
        filename="calibration_patches",
        data_path=data_path,
        time_resolution=time_resolution,
        duration=duration,
        antenna_mode=antenna_mode,
        basestation=basestation,
        reuse_tile_beam=True,
    )


def _DDEcal_iteration_station(gains, vis_station, coh_station, n_patches):
    m_chunk = coh_station * gains.T[:, np.newaxis, np.newaxis, :]

    # The single letter variables correspond to the matrix names in Gan et al. (2022)
    V = vis_station.swapaxes(0, 2).reshape(
        1, -1
    )  # time, freq, stat --> stat, freq * time
    M = m_chunk.swapaxes(1, 3).reshape(
        n_patches, -1
    )  # dir, time, freq, stat --> dir, stat, freq * time

    # M-1 = R-1Q-1 = R-1Q.T
    # V=JM = J (MT)T = J (QR).T
    # J = J (QR).T ((QR).T)-1 = V ((QR).T)-1 = V (R.TQ.T)-1 = V Q.T-1 R.T-1 = V Q R.T-1
    # Q, R = np.linalg.qr(M)
    # R_inv = solve_triangular(R, np.eye(n_patches), lower=False)
    pseudoinverse = np.conj(M).T @ np.linalg.inv(M @ np.conj(M).T)
    return V @ pseudoinverse, np.sum(np.abs(V - gains @ M))


def _DDEcal_smooth_frequencies(gains, frequencies, smoothness_scale):
    """
    Doesn't work for variable smoothing kernel or non-gaussian smoothing

    Parameters
    ----------
    gains : _type_
        _description_
    frequencies : _type_
        _description_
    smoothness_scale : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    if smoothness_scale == 0:
        return gains
    smoothed_gains = np.empty_like(gains)
    weights = np.ones(
        gains.shape
    )  # will be used for reweighting in later implementation

    for i, f in enumerate(frequencies):
        # Kernel based on relative spectral distance
        distances = (f - frequencies) / smoothness_scale
        mask = (-1 < distances) * (distances < 1)

        kernel = np.exp(-(distances[mask] ** 2) * 9)

        convolved_gains = np.sum(
            kernel[:, None, None] * weights[mask, :, :] * gains[mask, :, :], axis=0
        )
        convolved_weights = np.sum(kernel[:, None, None] * weights[mask, :, :], axis=0)

        convolved_weights[convolved_weights == 0] = (
            np.nan
        )  # can't smooth with zero weigths

        smoothed_gains[i, :, :] = convolved_gains / convolved_weights
    return smoothed_gains


def _DDEcal(
    vis_chunk,
    coh_chunk,
    smoothness_scale,
    frequencies,
    n_channels,
    stations,
    baselines,
    n_iterations,
    tolerance,
    update_speed,
):

    n_stations = len(stations)
    n_patches = coh_chunk.shape[0]
    n_freqs_total = coh_chunk.shape[2]
    n_spectral_sols = n_freqs_total // n_channels
    gains = np.ones([n_spectral_sols, n_stations, n_patches]).astype(complex)

    new_gains = np.zeros_like(gains)
    iteration = 0

    bl_mask = np.ones([n_stations, baselines.size // 2]).astype(bool)
    for i, station in enumerate(stations):
        bl_mask[i, :] = (baselines[:, 0] == station) | (baselines[:, 1] == station)
    residuals = np.zeros(n_iterations)
    while iteration < n_iterations and (
        np.sum(np.abs(new_gains - gains))  # conditie updaten!!!
        > tolerance * (1 - update_speed)
    ):
        for i, station in enumerate(stations):
            for f in range(n_spectral_sols):
                new_gains[f, i, :], new_residuals = _DDEcal_iteration_station(
                    gains=gains[f, :, :],
                    vis_station=vis_chunk[
                        :, f * n_channels : (f + 1) * n_channels, bl_mask[i, :]
                    ],
                    coh_station=coh_chunk[
                        :, :, f * n_channels : (f + 1) * n_channels, bl_mask[i, :]
                    ],
                    n_patches=n_patches,
                )
                residuals[iteration] += new_residuals
        gains = (1 - update_speed) * gains + update_speed * new_gains
        gains = _DDEcal_smooth_frequencies(
            gains=gains, frequencies=frequencies, smoothness_scale=smoothness_scale
        )
        iteration += 1
    residuals[iteration + 1 :] = np.nan
    return {"gains": gains, "residuals": residuals}


def read_data(visibility_file, array):
    with open(visibility_file) as csv_file:
        data = list(csv.DictReader(csv_file))
    frequencies = np.unique([float(row["frequency"]) for row in data])
    times = np.unique([row["time"] for row in data])
    time_information = get_time_info(times)
    baselines = np.unique([tuple(row["baseline"].split("-")) for row in data], axis=0)
    stations = np.unique([row["baseline"].split("-")[0] for row in data])
    baseline_length = calculate_baseline_length(array, baselines)
    visibilities = [complex(row["visibility"]) for row in data]
    visibilities = np.array(visibilities).reshape(
        times.size, frequencies.size, baselines.size // 2
    )
    return (
        visibilities,
        times,
        time_information,
        frequencies,
        stations,
        baselines,
        baseline_length,
    )


def calculate_baseline_length(array, baselines):
    p1 = np.array([array[station].p for station in baselines[:, 0]])
    p2 = np.array([array[station].p for station in baselines[:, 1]])
    distance = np.linalg.norm(p1 - p2, axis=1)
    return distance


def get_time_info(times):
    t1 = Time(times[0])
    t2 = Time(times[1])
    t_end = Time(times[-1])

    dt = t2 - t1
    time_band = (
        t_end - t1
    ) + dt  # one extra, to account for the half timestep a before t1 and after t_end

    info = {"time_resolution": dt.sec, "duration": time_band.sec / 3600}
    return info


def preflagger(frequencies, uv_lambda, visibilities, baseline_length):
    min_l = const.c.value / frequencies * uv_lambda[0]
    max_l = const.c.value / frequencies * uv_lambda[1]

    too_short = baseline_length[np.newaxis, :] < min_l[:, np.newaxis]
    too_long = baseline_length[np.newaxis, :] > max_l[:, np.newaxis]
    flagmask = too_short | too_long

    return flagmask


def run_DDEcal(
    array,
    reference_station,
    skymodel,
    visibility_file,
    n_channels=1,
    n_times=1,
    uv_lambda=[250, 5000],
    reuse_predict=False,
    antenna_mode="omnidirectional",
    n_iterations=50,
    tolerance=1e-3,
    update_speed=0.5,
    smoothness_scale=4e6,
):
    data_path = visibility_file.rstrip("/data.csv")
    # parse visibility

    (
        visibilities,
        times,
        time_information,
        frequencies,
        stations,
        baselines,
        baseline_length,
    ) = read_data(visibility_file, array)

    flag_mask = preflagger(frequencies, uv_lambda, visibilities, baseline_length)
    visibilities[:, flag_mask] = 0

    # t,bl,f
    # 1 chunk en dat alle t,f dan bl
    if not reuse_predict:
        predict_model(
            array,
            skymodel,
            frequencies,
            times[0],
            visibility_file,
            time_information["time_resolution"],
            time_information["duration"],
            antenna_mode,
            reference_station,
        )

    patch_names = skymodel.elements.keys()
    n_patches = len(patch_names)
    coherency = []
    for patch_name in patch_names:
        patch_file = f"{data_path}/calibration_patches/patch_models/{patch_name}.csv"
        patch_coherency = read_data(patch_file, array)[0]
        coherency.append(patch_coherency)
    coherency = np.array(coherency)
    if n_patches == 1:
        coherency.reshape(1, len(times), len(frequencies), baselines.size // 2)
    coherency[:, :, flag_mask] = 0

    # for t in np.arange(0, times.size, n_times):
    #     results = _DDEcal(
    #         vis_chunk=visibilities[t : t + n_times, :, :],
    #         coh_chunk=coherency[:, t : t + n_times, :, :],
    #         smoothness_scale=smoothness_scale,
    #         frequencies=frequencies,
    #         n_channels=n_channels,
    #         stations=stations,
    #         baselines=baselines,
    #         n_iterations=n_iterations,
    #         tolerance=tolerance,
    #         update_speed=update_speed,
    #     )

    results = Parallel(n_jobs=-1)(
        delayed(_DDEcal)(
            vis_chunk=visibilities[t : t + n_times, :, :],
            coh_chunk=coherency[:, t : t + n_times, :, :],
            smoothness_scale=smoothness_scale,
            frequencies=frequencies,
            n_channels=n_channels,
            stations=stations,
            baselines=baselines,
            n_iterations=n_iterations,
            tolerance=tolerance,
            update_speed=update_speed,
        )
        for t in np.arange(0, times.size, n_times)
    )

    return results
    # os.makedirs(f"{data_path}/ddecal", exists_ok=True)
    # with open(f"{data_path}/ddecal/convergence.csv", "w") as csvfile:
    #     writer = csv.writer(csvfile, delimiter=",")
    #     writer.writerow("iteration", "loss")
    #     [writer.writerow(i, result) for i, result in enumerate(results)]

    # with open(f"{data_path}/ddecal/convergence.csv", "w") as csvfile:
    #     writer = csv.writer(csvfile, delimiter=",")
    #     writer.writerow("iteration", "loss")
    #     [writer.writerow(i, result) for i, result in enumerate(results)]

    # parallelize over time
    # alle fsequential per iteratie, dan smoother
    # creer plots loss & gains
    # stap over op codex
    # grote run
    # nieuwe smoother
    # grote run
    # sagecal
