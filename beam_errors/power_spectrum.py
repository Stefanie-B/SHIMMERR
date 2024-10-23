import csv
import numpy as np
import tqdm
from astropy.time import Time
import os
from astropy.constants import c
from joblib import Parallel, delayed


class delay_spectrum:
    def __init__(self, array, reference_station):
        self.array = array
        self.reference_station = self.array[reference_station]

    def _read_metadata(self, file):
        with open(file) as csv_file:
            data = list(csv.DictReader(csv_file))

            self.frequencies = np.unique([float(row["frequency"]) for row in data])
            self.n_freqs = len(self.frequencies)
            self.delay = np.fft.fftfreq(
                self.n_freqs, d=self.frequencies[1] - self.frequencies[0]
            )

            self.times = np.unique([row["time"] for row in data])
            self.n_times = len(self.times)
            if self.n_times > 1:
                self.time_resolution = (Time(self.times[1]) - Time(self.times[0])).sec

            self.baselines = np.unique(
                [tuple(row["baseline"].split("-")) for row in data], axis=0
            )
            self.n_baselines = self.baselines.size // 2

    def _read_single_file(self, visibility_file):
        with open(visibility_file) as csv_file:
            data = csv.DictReader(csv_file)
            visibilities = [complex(row["visibility"]) for row in data]

        visibilities = np.array(visibilities).reshape(
            -1, self.n_freqs, self.n_baselines
        )
        return visibilities

    def _compute_auto_power(self, data):
        return abs(data) ** 2

    def _compute_cross_power(self, data_1, data_2):
        return data_1 * np.conj(data_2)

    def _calculate_baseline_lengths(self):
        pointing_direction = self.reference_station.radec_to_ENU(
            self.times[0],
            temporal_offset=self.time_resolution,
            number_of_timesteps=self.n_times,
            tracking_direction=True,
        )
        [
            station.set_array_position(self.reference_station)
            for station in self.array.values()
        ]
        # Je kunt de baselines berekenen en dan de pointing directions eruit projecteren
        # dit neemt UV rotaties niet mee, maar die maken ook niet uit, want we kijken naar lengte

        baseline_set = np.empty([self.n_times, self.n_baselines])
        for baseline_number, (station_1, station_2) in enumerate(self.baselines):
            baseline_vector = (
                self.array[station_1].p_array - self.array[station_2].p_array
            )
            projected_baseline = (
                baseline_vector[:, np.newaxis]
                - np.dot(baseline_vector, pointing_direction) * pointing_direction
            )
            baseline_set[:, baseline_number] = np.linalg.norm(
                projected_baseline[:2, :], axis=0
            )
        # baseline_set = (
        #     baseline_set[:, np.newaxis, :]
        #     * self.frequencies[np.newaxis, :, np.newaxis]
        #     / c.value
        # )  # the order of the axes is the same as the d  ata for consistency, even if it requires swapping the axes later on
        return baseline_set

    def _assign_baseline_bin(self, data, baseline_bins):
        # get baseline lengths and bins
        baseline_lengths = self._calculate_baseline_lengths()

        # reshape data
        data = data.swapaxes(0, 1).reshape(self.n_freqs, -1)
        baseline_lengths = baseline_lengths.reshape(-1)

        # bin the baselines
        bin_indices = np.digitize(baseline_lengths, bins=baseline_bins) - 1

        # prepare the data arrays
        binned_data = np.zeros([self.n_freqs, len(baseline_bins) - 1], dtype=complex)
        weights = np.zeros([self.n_freqs, len(baseline_bins) - 1])

        # mask out data outside bins
        valid_mask = (bin_indices >= 0) & (bin_indices < len(baseline_bins) - 1)

        # bin the data
        [
            np.add.at(binned_data[i], bin_indices[valid_mask], data[i][valid_mask])
            for i in range(self.n_freqs)
        ]
        weights = np.bincount(bin_indices[valid_mask], minlength=len(baseline_bins) - 1)
        return binned_data, weights

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

    def _grid_file(self, file, baseline_bins):
        data = self._read_single_file(file)
        ft_data = np.fft.fft(data, axis=1, norm="ortho")
        new_bin, new_weights = self._assign_baseline_bin(ft_data, baseline_bins)
        return {"data": new_bin, "weights": new_weights}

    def _grid_data(self, visibility_files, baseline_bins):
        results = Parallel(n_jobs=-1)(
            delayed(self._grid_file)(
                file=file,
                baseline_bins=baseline_bins,
            )
            for file in visibility_files
        )
        weights = sum([result["weights"] for result in results])
        data = sum([result["data"] for result in results]) / weights[np.newaxis, :]

        return data, weights

    def calculate_delay_spectrum(
        self, files, savename, baseline_bins, cross_files=None
    ):
        self._read_metadata(files[0])
        data, weights = self._grid_data(files, baseline_bins)
        if cross_files is None:
            spectrum = self._compute_auto_power(data)
        else:
            cross_data, cross_weights = self._grid_data(cross_files, baseline_bins)
            spectrum = self._compute_cross_power(data, cross_data)
            weights += cross_weights

        self._write_PS(savename, baseline_bins, spectrum, weights)

    # load data
    # FT along freq
    # calculate (auto or cross)
    # bin
    ## calc bl lengths
    # save
