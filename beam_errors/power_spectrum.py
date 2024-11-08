import csv
import numpy as np
import tqdm
from astropy.time import Time
import os
from astropy.constants import c
from joblib import Parallel, delayed
from scipy.stats import binned_statistic_2d


class delay_spectrum:
    def __init__(self, array, reference_station):
        self.array = array
        self.reference_station = self.array[reference_station]

    def _read_metadata(self, file):
        with open(file) as csv_file:
            data = list(csv.DictReader(csv_file))

            self.frequencies = np.unique([float(row["frequency"]) for row in data])
            self.n_freqs = len(self.frequencies)
            self.delay = np.fft.fftshift(
                np.fft.fftfreq(
                    self.n_freqs, d=self.frequencies[1] - self.frequencies[0]
                )
            )

            self.baselines = np.unique(
                [tuple(row["baseline"].split("-")) for row in data], axis=0
            )
            self.n_baselines = self.baselines.size // 2

    def _read_single_file(self, visibility_file):
        with open(visibility_file) as csv_file:
            data = list(csv.DictReader(csv_file))

        times = np.unique([row["time"] for row in data])

        visibilities = [complex(row["visibility"]) for row in data]
        visibilities = np.array(visibilities).reshape(
            -1, self.n_freqs, self.n_baselines
        )
        return times, visibilities

    def _compute_auto_power(self, data):
        power = abs(data) ** 2
        return np.mean(power, axis=-1)

    def _compute_cross_power(self, data_1, data_2):
        power = data_1 * np.conj(data_2)
        return np.mean(power, axis=-1)

    @staticmethod
    def _rotate_baseline(k, v, theta):
        # Rodrigues' rotatio formula
        v_rotated = (
            v * np.cos(theta)
            + np.linalg.cross(k.T, v.T).T * np.sin(theta)
            + k * np.diag(k.T @ v) * (1 - np.cos(theta))
        )
        return v_rotated

    def _calculate_uv_plane(self, times):
        n_times = len(times)
        time_objects = [Time(time) for time in times]
        if n_times > 1:
            time_resolution = (time_objects[1] - time_objects[0]).sec
        else:
            time_resolution = 0

        pointing_direction = self.reference_station.radec_to_ENU(
            times[0],
            temporal_offset=time_resolution,
            number_of_timesteps=len(times),
            tracking_direction=True,
        )

        [
            station.set_array_position(self.reference_station)
            for station in self.array.values()
        ]
        # Je kunt de baselines berekenen en dan de pointing directions eruit projecteren
        # dit neemt UV rotaties niet mee, maar die maken ook niet uit, want we kijken naar lengte

        julian_dates = np.array([t.jd - 2451545.0 for t in time_objects])
        ERA = 2 * np.pi * (0.7790572732640 + 1.00273781191135448 * julian_dates)

        baseline_set = np.empty([2, n_times, self.n_baselines])
        for baseline_number, (station_1, station_2) in enumerate(self.baselines):
            baseline_vector = (
                self.array[station_1].p_array - self.array[station_2].p_array
            )

            projected_baseline = baseline_vector[:, np.newaxis] - pointing_direction * (
                baseline_vector[np.newaxis, :] @ pointing_direction
            )

            rotated_baseline = self._rotate_baseline(
                pointing_direction, projected_baseline, ERA
            )

            baseline_set[0, :, baseline_number] = np.linalg.norm(
                rotated_baseline[:2, :], axis=0
            )

            baseline_set[1, :, baseline_number] = np.arctan2(
                rotated_baseline[1], rotated_baseline[0]
            ) % (2 * np.pi)
        # baseline_set = (
        #     baseline_set[:, np.newaxis, :]
        #     * self.frequencies[np.newaxis, :, np.newaxis]
        #     / c.value
        # )  # the order of the axes is the same as the d  ata for consistency, even if it requires swapping the axes later on
        return baseline_set

    def _digitize_baseline_set(baseline_set, baseline_bins, angular_resolution):
        lengths = np.reshape(baseline_set[0, ...]).reshape(-1)
        length_indices = np.digitize(lengths, bins=baseline_bins)

        angular_bins = np.arange(-np.pi, np.pi, angular_resolution / 180 * np.pi)
        angles = np.reshape(baseline_set[1, ...]).reshape(-1)
        angle_indices = np.digitize(angles, bins=angular_bins)
        return length_indices, angle_indices

    def _assign_baseline_bin(self, data, times, baseline_bins, angular_resolution):
        # get baseline lengths and bins
        baseline_positions = self._calculate_uv_plane(times)

        angular_bins = np.arange(0, 2 * np.pi + 1e-6, angular_resolution / 180 * np.pi)

        lengths = baseline_positions[0, ...].reshape(-1)
        angles = baseline_positions[1, ...].reshape(-1)

        data = data.swapaxes(0, 1).reshape(self.n_freqs, -1)

        # mask out data outside bins
        valid_mask = (lengths >= baseline_bins[0]) & (lengths < baseline_bins[-1])
        lengths = lengths[valid_mask]
        angles = angles[valid_mask]
        data = data[:, valid_mask]

        bin_sum = np.array(
            [
                binned_statistic_2d(
                    x=lengths,
                    y=angles,
                    values=data[f, :],
                    statistic="sum",
                    bins=[baseline_bins, angular_bins],
                )[0]
                for f in range(self.n_freqs)
            ]
        )
        bin_sum = np.nan_to_num(bin_sum, nan=0.0)

        bin_weight = np.array(
            binned_statistic_2d(
                x=lengths,
                y=angles,
                values=data[0, :],
                statistic="count",
                bins=[baseline_bins, angular_bins],
            )[0]
        )

        # Add the complex conjugates of the visibilities (count both baseline i,j and j,i)
        valid_mask = (
            (lengths >= baseline_bins[0])
            & (lengths < baseline_bins[-1])
            & (lengths > 0)
        )  # excludes autocorrelations
        lengths = lengths[valid_mask]
        angles = angles[valid_mask]
        data = np.conj(data[:, valid_mask])
        angles = (angles + np.pi) % (2 * np.pi)

        bin_sum_conj = np.array(
            [
                binned_statistic_2d(
                    x=lengths,
                    y=angles,
                    values=data[f, :],
                    statistic="sum",
                    bins=[baseline_bins, angular_bins],
                )[0]
                for f in range(self.n_freqs)
            ]
        )
        bin_sum_conj = np.nan_to_num(bin_sum_conj, nan=0.0)

        bin_weight_conj = np.array(
            binned_statistic_2d(
                x=lengths,
                y=angles,
                values=data[0, :],
                statistic="count",
                bins=[baseline_bins, angular_bins],
            )[0]
        )

        total_sum = bin_sum + bin_sum_conj
        total_weight = bin_weight + bin_weight_conj
        return total_sum, total_weight

    def _write_PS(self, savename, baseline_bins, spectrum, weights):
        os.makedirs("/".join(savename.split("/")[:-1]), exist_ok=True)
        with open(savename, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["UV bin start", "UV bin end", "delay", "value"])
            dataset = [
                [
                    baseline_bins[:-1][b],
                    baseline_bins[1:][b],
                    self.delay[f],
                    spectrum[f, b],
                ]
                for b in range(len(baseline_bins) - 1)
                for f in range(self.n_freqs)
            ]
            writer.writerows(dataset)
        name = savename.rstrip(".csv")
        with open(f"{name}_weights.csv", "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["UV bin start", "UV bin end", "value"])
            dataset = [
                [
                    baseline_bins[:-1][b],
                    baseline_bins[1:][b],
                    weights[b],
                ]
                for b in range(len(baseline_bins) - 1)
            ]
            writer.writerows(dataset)

    def _grid_file(self, file, baseline_bins, angular_resolution):
        times, data = self._read_single_file(file)
        ft_data = np.fft.fftshift(np.fft.fft(data, axis=1, norm="ortho"), axes=1)
        new_bin, new_weights = self._assign_baseline_bin(
            ft_data, times, baseline_bins, angular_resolution
        )
        return {"data": new_bin, "weights": new_weights}

    def _grid_data(self, visibility_files, baseline_bins, angular_resolution):
        results = Parallel(n_jobs=10)(
            delayed(self._grid_file)(
                file=file,
                baseline_bins=baseline_bins,
                angular_resolution=angular_resolution,
            )
            for file in visibility_files
        )
        # results = []
        # for file in visibility_files:
        #     results.append(
        #         self._grid_file(
        #             file=file,
        #             baseline_bins=baseline_bins,
        #             angular_resolution=angular_resolution,
        #         )
        #     )

        weights = sum([result["weights"] for result in results])
        data = sum([result["data"] for result in results]) / weights[np.newaxis, :, :]
        data[:, weights == 0] = 0
        return data, np.sum(weights, axis=-1)

    def calculate_delay_spectrum(
        self, files, savename, baseline_bins, cross_files=None, angular_resolution=1
    ):
        self._read_metadata(files[0])
        data_2D, weights = self._grid_data(files, baseline_bins, angular_resolution)
        if cross_files is None:
            spectrum = self._compute_auto_power(data_2D)
        else:
            cross_data_2D, cross_weights = self._grid_data(
                cross_files, baseline_bins, angular_resolution
            )
            spectrum = self._compute_cross_power(data_2D, cross_data_2D)
            weights += cross_weights

        self._write_PS(savename, baseline_bins, spectrum, weights)

    # load data
    # FT along freq
    # calculate (auto or cross)
    # bin
    ## calc bl lengths
    # save
