import copy
from beam_errors.visibility import predict_patch_visibilities
import csv
import numpy as np
from astropy import constants as const
from astropy.time import Time
from joblib import Parallel, delayed
import pickle
import os
from numba import jit


class DDEcal:
    def __init__(
        self,
        array,
        reference_station,
        n_channels=1,
        n_times=1,
        uv_lambda=[250, 5000],
        antenna_mode="omnidirectional",
        n_iterations=50,
        tolerance=1e-6,
        update_speed=0.2,
        smoothness_scale=4e6,
    ):
        self.array = array
        self.n_stations = len(array)
        self.reference_station = reference_station
        self.n_freqs_per_sol = n_channels
        self.n_times_per_sol = n_times
        self.uv_lambda = uv_lambda
        self.antenna_mode = antenna_mode
        self.n_iterations = n_iterations
        self.tolerance = tolerance
        self.update_speed = update_speed
        self.smoothness_scale = smoothness_scale

        if not 0 < self.update_speed <= 1:
            raise ValueError(
                f"An update speed of {update_speed} is not permissible. Please choose a value larger than 0 and less than or equal to 1."
            )
        elif self.update_speed == 1:
            self._change_rate = lambda gains, new_gains: np.mean(
                np.abs(new_gains - gains)
            )
        else:
            self._change_rate = lambda gains, new_gains: np.mean(
                np.abs(new_gains - gains)
            ) / (1 - self.update_speed)

    def _set_time_info(self):
        t1 = Time(self.times[0])
        t2 = Time(self.times[1])
        t_end = Time(self.times[-1])

        dt = t2 - t1
        time_band = (
            (t_end - t1) + dt
        )  # one extra, to account for the half timestep a before t1 and after t_end

        self.time_resolution = dt.sec
        self.duration = time_band.sec / 3600

    def _set_baseline_length(self):
        p1 = np.array([self.array[station].p for station in self.baselines[:, 0]])
        p2 = np.array([self.array[station].p for station in self.baselines[:, 1]])
        distance = np.linalg.norm(p1 - p2, axis=1)
        self.baseline_length = distance

    def _read_data(self, visibility_file, update_metadata=True):
        with open(visibility_file) as csv_file:
            data = list(csv.DictReader(csv_file))
        if update_metadata:
            self.frequencies = np.unique([float(row["frequency"]) for row in data])
            self.n_freqs = len(self.frequencies)
            self.times = np.unique([row["time"] for row in data])
            self.n_times = len(self.times)
            self._set_time_info()
            self.baselines = np.unique(
                [tuple(row["baseline"].split("-")) for row in data], axis=0
            )
            self.n_baselines = self.baselines.size // 2
            self._set_baseline_length()
            self.stations = np.unique([row["baseline"].split("-")[0] for row in data])

        visibilities = [complex(row["visibility"]) for row in data]
        visibilities = np.array(visibilities).reshape(
            self.n_times, self.n_freqs, self.n_baselines
        )
        return visibilities

    def _run_preflagger(self):
        min_l = const.c.value / self.frequencies * self.uv_lambda[0]
        max_l = const.c.value / self.frequencies * self.uv_lambda[1]

        too_short = self.baseline_length[np.newaxis, :] < min_l[:, np.newaxis]
        too_long = self.baseline_length[np.newaxis, :] > max_l[:, np.newaxis]
        self.flag_mask = too_short | too_long

    def _predict_model(self, skymodel):
        unit_gain_array = copy.deepcopy(self.array)
        for station in unit_gain_array.values():
            station.reset_elements()
            station.g = 1.0 + 0j

        predict_patch_visibilities(
            array=unit_gain_array,
            skymodel=skymodel,
            frequencies=self.frequencies,
            start_time_utc=self.times[0],
            filename="calibration_patches",
            data_path=self.data_path,
            time_resolution=self.time_resolution,
            duration=self.duration,
            antenna_mode=self.antenna_mode,
            basestation=self.reference_station,
            reuse_tile_beam=True,
        )

    def _DDEcal_station_iteration(self, gains, visibility, coherency):
        m_chunk = coherency * np.conj(gains.T[:, np.newaxis, np.newaxis, :])

        # The single letter variables correspond to the matrix names in Gan et al. (2022)
        # We swap the temporal and station axes to get station to the front and then flatten to the desired shapes
        V = visibility.swapaxes(0, 2).reshape(1, -1)
        M = m_chunk.swapaxes(1, 3).reshape(self.n_patches, -1)

        # Remove flagged data
        mask = np.squeeze(~np.isnan(V))
        V = V[:, mask]
        M = M[:, mask]

        # Solve V = JM
        new_gains, loss = np.linalg.lstsq(M.T, V.T, rcond=-1)[:2]
        last_residual = np.sum(np.abs(V - gains @ M) ** 2)
        return np.squeeze(new_gains), last_residual, np.squeeze(loss)

    def _DDEcal_smooth_frequencies(self, gains, weights):
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
        if self.smoothness_scale == 0:
            return gains

        smoothed_gains = np.empty_like(gains)

        for i, f in enumerate(self.frequencies):
            # Kernel based on relative spectral distance
            distances = (f - self.frequencies) / self.smoothness_scale
            mask = (-1 < distances) * (distances < 1)

            # Gaussian kernel
            kernel = np.exp(-(distances[mask] ** 2) * 9)

            convolved_gains = np.sum(
                kernel[:, None, None] * weights[mask, :, :] * gains[mask, :, :], axis=0
            )
            convolved_weights = np.sum(
                kernel[:, None, None] * weights[mask, :, :], axis=0
            )

            # convolved_weights[convolved_weights == 0] = (
            #     np.nan
            # )  # can't smooth with zero weigths
            if np.sum(convolved_weights == 0) > 0:
                print(
                    f"There are now {np.sum(convolved_weights==0)} ill-defined weights, replacing the gains with zeros"
                )
                convolved_gains[np.where(convolved_weights == 0)] = 0
                convolved_weights[np.where(convolved_weights == 0)] = 1

            smoothed_gains[i, :, :] = convolved_gains / convolved_weights
        return smoothed_gains

    def _remove_unitairy_ambiguity(self, gains):
        amplitudes = np.abs(gains)
        phases = np.angle(gains)

        reference_phase = np.squeeze(
            phases[:, self.stations == self.reference_station, :], axis=1
        )
        corrected_phases = phases - reference_phase[:, np.newaxis, :]

        corrected_gains = amplitudes * np.exp(1j * corrected_phases)
        return corrected_gains

    def _select_baseline_data(self, data, station_number):
        selected_data = data[..., self.baseline_mask[station_number, :]]
        selected_data[..., :station_number] = np.conj(
            selected_data[..., :station_number]
        )
        return selected_data

    def _DDEcal_timeslot(self, visibility, coherency):
        gains = np.ones([self.n_spectral_sols, self.n_stations, self.n_patches]).astype(
            complex
        )

        new_gains = np.zeros_like(gains)
        iteration = 0

        residuals = np.zeros(self.n_iterations)
        loss = np.zeros(self.n_iterations)

        while (
            iteration < self.n_iterations
            and self._change_rate(gains, new_gains) > self.tolerance
        ):
            weights = np.zeros_like(gains, dtype=float)
            for i, station in enumerate(self.stations):
                selected_visibilities = self._select_baseline_data(visibility, i)
                selected_coherencies = self._select_baseline_data(coherency, i)
                for f in range(self.n_spectral_sols):
                    # Give the visibilities for the frequencies in this slot and the baselines connected to this station
                    selected_frequencies = range(
                        f * self.n_freqs_per_sol, (f + 1) * self.n_freqs_per_sol
                    )
                    new_gains[f, i, :], new_residual, new_loss = (
                        self._DDEcal_station_iteration(
                            gains=gains[f, :, :],
                            visibility=np.take(
                                selected_visibilities,
                                indices=selected_frequencies,
                                axis=1,
                            ),
                            coherency=np.take(
                                selected_coherencies,
                                indices=selected_frequencies,
                                axis=2,
                            ),
                        )
                    )
                    residuals[iteration] += new_residual
                    loss[iteration] += new_loss
                    weights[f, i, :] = self._reweight_function(selected_coherencies)

            gain_update = (
                1 - self.update_speed
            ) * gains + self.update_speed * new_gains

            gains = self._DDEcal_smooth_frequencies(
                gains=gain_update,
                weights=weights,
            )

            iteration += 1
        gains = self._remove_unitairy_ambiguity(gains)
        residuals /= visibility.size
        return {
            "gains": gains,
            "residuals": residuals[:iteration],
            "loss": loss[:iteration],
            "n_iter": iteration,
        }

    def run_DDEcal(
        self,
        visibility_file,
        skymodel,
        reuse_predict=False,
        reweight_mode=None,
        fname=None,
    ):
        self.data_path = "/".join(visibility_file.split("/")[:-1])

        if reweight_mode == "abs":
            self._reweight_function = lambda coherency: np.sum(
                np.abs(coherency), axis=(1, 2, 3)
            )
        elif reweight_mode == "squared":
            self._reweight_function = lambda coherency: np.sum(
                np.abs(coherency) ** 2, axis=(1, 2, 3)
            )
        elif reweight_mode is None or reweight_mode == "none":
            self._reweight_function = lambda coherency: np.ones(coherency.shape[0])
        else:
            raise ValueError("Invalid reweight mode")

        # parse visibility
        visibilities = self._read_data(visibility_file, True)

        self._run_preflagger()
        visibilities[:, self.flag_mask] = np.nan

        # t,bl,f
        # 1 chunk en dat alle t,f dan bl
        if not reuse_predict:
            self._predict_model(skymodel)

        patch_names = skymodel.elements.keys()
        self.n_patches = len(patch_names)
        coherency = []
        for patch_name in patch_names:
            patch_file = (
                f"{self.data_path}/calibration_patches/patch_models/{patch_name}.csv"
            )
            patch_coherency = self._read_data(patch_file, False)
            coherency.append(patch_coherency)
        coherency = np.array(coherency)
        if self.n_patches == 1:
            coherency.reshape(1, self.n_times, self.n_freqs, self.n_baselines)
        coherency[:, :, self.flag_mask] = np.nan

        # select which rows of the visibility matrix will be active for each station (baseline mask)
        self.baseline_mask = np.ones([self.n_stations, self.n_baselines]).astype(bool)
        for i, station in enumerate(self.stations):
            self.baseline_mask[i, :] = (self.baselines[:, 0] == station) | (
                self.baselines[:, 1] == station
            )

        # Set the number of spectral solutions
        self.n_spectral_sols = self.n_freqs // self.n_freqs_per_sol
        if self.n_spectral_sols * self.n_freqs_per_sol < self.n_freqs:
            self.n_spectral_sols += 1  # last frequency slot has smaller set of data

        results = Parallel(n_jobs=-1)(
            delayed(self._DDEcal_timeslot)(
                visibility=visibilities[t : t + self.n_times_per_sol, :, :],
                coherency=coherency[:, t : t + self.n_times_per_sol, :, :],
            )
            for t in np.arange(0, self.n_times, self.n_times_per_sol)
        )

        metadata = {
            "smoothness_scale": self.smoothness_scale,
            "times": [
                self.times[t] for t in np.arange(0, self.n_times, self.n_times_per_sol)
            ],
            "frequencies": np.array(
                [
                    np.mean(
                        self.frequencies[
                            i * self.n_freqs_per_sol : (i + 1) * self.n_freqs_per_sol
                        ]
                    )
                    for i in range(self.n_spectral_sols)
                ]
            ),
            "directions": list(patch_names),
            "stations": self.stations,
        }

        if fname is None:
            fname = f"Calibration_results_{reweight_mode}_{int(self.smoothness_scale)}"

        os.makedirs(f"{self.data_path}/calibration_results/", exist_ok=True)
        with open(f"{self.data_path}/calibration_results/{fname}", "wb") as fp:
            pickle.dump(results, fp)
        with open(f"{self.data_path}/calibration_results/{fname}_metadata", "wb") as fp:
            pickle.dump(metadata, fp)

        return results
