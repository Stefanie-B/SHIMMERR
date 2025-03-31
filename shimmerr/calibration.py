import copy
from shimmerr.visibility import predict_patch_visibilities
import csv
import numpy as np
from astropy import constants as const
from astropy.time import Time
from joblib import Parallel, delayed
import pickle
import os


class DDEcal:
    """
    A class that can run DDECal in a single polarisation, with various weighting schemes.

    Attributes:
    -------------
    array: dict
        Interferometer dictionary with Station objects as values.
    reference_station: str
        Name (AKA array key) of the station that should act as the reference for gain phases. Subtracting the phases of this station removes unitairy ambiguity.
    n_channels: int
        Number of channels per solution interval
    n_times: int
        Number of time steps per solution interval
    uv_lambda: list
        List containing the minimum and maximum baseline length used in calibration in lambda.
    antenna_mode: str
        Mode for the antenna in the calibration beam (see: Antenna objects).
    n_iterations: int
        Number of iterations for the algorithm.
    tolerance: float
        If the mean change rate of the gains drops below this number, the algorithm interprets it as convergence and stops the optimisation earlier than n_iterations.
    update_spead: float
        Number between 0 and 1 that determines how quick the gain solutions should update. A speed of 0 means no update, a speed of 1 means that the previous gain solution is not included in the update.
    smoothness_scale: float
        Width of the spectral regularisation kernel in MHz.
    """

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
        """
        Sets the time parameters (resolution and duration)
        """
        t1 = Time(self.times[0])
        t2 = Time(self.times[1])
        t_end = Time(self.times[-1])

        dt = t2 - t1
        time_band = (
            t_end - t1
        ) + dt  # one extra, to account for the half timestep a before t1 and after t_end

        self.time_resolution = dt.sec
        self.duration = time_band.sec / 3600

    def _set_baseline_length(self):
        """
        Calculates the lengths of all baselines based on the station positions in the reference stations ENU frame
        """
        [
            station.set_array_position(self.array[self.reference_station])
            for station in self.array.values()
        ]
        p1 = np.array([self.array[station].p_array for station in self.baselines[:, 0]])
        p2 = np.array([self.array[station].p_array for station in self.baselines[:, 1]])
        baseline = p1 - p2

        # Calculate the phase center direction in the reference station's ENU system
        phase_center = (
            self.array[self.reference_station]
            .radec_to_ENU(
                time=self.times[0],
                temporal_offset=self.time_resolution,
                number_of_timesteps=self.n_times,
                tracking_direction=True,
            )
            .T
        )

        # Calculate the projected baseline lengths
        parallel_component = np.dot(phase_center, baseline.T)
        parallel_component_vector = (
            parallel_component[:, :, np.newaxis] * phase_center[:, np.newaxis, :]
        )
        perpendicular_component = baseline[np.newaxis, ...] - parallel_component_vector
        projected_baseline = np.linalg.norm(perpendicular_component, axis=-1)

        self.baseline_length = projected_baseline

    def _read_data(self, visibility_file, update_metadata=True):
        """
        Reads a csv of visibilities and parses it.

        Parameters
        ----------
        visibility_file : str
            path + filename of the csv file with visibilties (in SHIMMERR format)
        update_metadata : bool, optional
            If True, sets all metadata (otherwise previously stored values are used), by default True

        Returns
        -------
        ndarray
            numpy array with the complex visibilities (time x frequency x baselines)
        """
        with open(visibility_file) as csv_file:
            data = list(csv.DictReader(csv_file))

        if update_metadata:
            self.frequencies = np.unique([float(row["frequency"]) for row in data])
            self.n_freqs = len(self.frequencies)

            # Set the number of spectral solutions
            self.n_spectral_sols = self.n_freqs // self.n_freqs_per_sol
            if self.n_spectral_sols * self.n_freqs_per_sol < self.n_freqs:
                self.n_spectral_sols += 1  # last frequency slot has smaller set of data

            # Set time info
            self.times = np.unique([row["time"] for row in data])
            self.n_times = len(self.times)
            self._set_time_info()
            self.baselines = np.unique(
                [tuple(row["baseline"].split("-")) for row in data], axis=0
            )

            # Set station info
            self.n_baselines = self.baselines.size // 2
            self._set_baseline_length()
            self.stations = np.unique([row["baseline"].split("-")[0] for row in data])

        # read visibilities
        visibilities = [complex(row["visibility"]) for row in data]
        visibilities = np.array(visibilities).reshape(
            self.n_times, self.n_freqs, self.n_baselines
        )
        return visibilities

    def _run_preflagger(self):
        """
        Preflags data based on baseline length (creates a flag mask that indicates which should be taken into account)
        """
        min_l = const.c.value / self.frequencies * self.uv_lambda[0]
        max_l = const.c.value / self.frequencies * self.uv_lambda[1]

        too_short = (
            self.baseline_length[:, np.newaxis, :] < min_l[np.newaxis, :, np.newaxis]
        )
        too_long = (
            self.baseline_length[:, np.newaxis, :] > max_l[np.newaxis, :, np.newaxis]
        )
        self.flag_mask = too_short | too_long

    def _predict_model(self, skymodel):
        """
        Does a forward predict of the model, assuming the stations are unperturbed to construct the coherency matrix

        Parameters
        ----------
        skymodel : SHIMMERR Skymodel object
            skymodel (see sources.py)
        """
        # Create an array without perturbations
        unit_gain_array = copy.deepcopy(self.array)
        for station in unit_gain_array.values():
            station.reset_elements()
            station.g = 1.0 + 0j

        # Create predictions
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

    def _DDEcal_station_iteration(self, gains, visibility, coherency, station_number):
        """
        Performs the least-squares fit in DDECal for 1 station at 1 solution interval
        (i.e. finds J in V = JM)

        Parameters
        ----------
        gains : ndarray
            gains of the previous iteration
        visibility : ndarray
            visibilities for baselines with this station (t x f x bl)
        coherency : ndarray
            coherency matrices to fit (patches x t x f x bl)
        station_number : int
            index of the station that is fitted

        Returns
        -------
        ndarray, float, float
            - least squares gain solution for this station (one gain for each patch)
            - residual of the previous iteration (cheaper than producing that of this iteration, because of the smoothing that happens later)
            - loss (residual of V - JM with the least squares solution for this iteration)
        """
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
        last_residual = np.sum(np.abs(V - gains[station_number, :] @ M) ** 2)

        return np.squeeze(new_gains), last_residual, np.sum(loss)

    def _DDEcal_smooth_frequencies(self, gains, weights):
        """
        Performs the intra-iteration regularisation.
        Doesn't work for variable smoothing kernel or non-gaussian smoothing

        Parameters
        ----------
        gains : ndarray of complex
            gains (frequency x station x direction patch)
        frequencies : ndarray of floats
            channel frequencies
        smoothness_scale : float
            standard deviation of the truncated Gaussian smoothing kernel

        Returns
        -------
        ndarray of complex
            smoothed gains
        """
        # Don't smooth if the scale is zero
        if self.smoothness_scale == 0:
            return gains

        smoothed_gains = np.empty_like(gains)
        for i, f in enumerate(self.frequencies):
            # Kernel based on relative spectral distance
            distances = (f - self.frequencies) / self.smoothness_scale
            mask = (-1 < distances) * (
                distances < 1
            )  # truncate (distance=1 is 3 sigma)

            # Gaussian kernel
            kernel = np.exp(-(distances[mask] ** 2) * 9)

            # do the convolutions
            convolved_gains = np.sum(
                kernel[:, None, None] * weights[mask, :, :] * gains[mask, :, :], axis=0
            )
            convolved_weights = np.sum(
                kernel[:, None, None] * weights[mask, :, :], axis=0
            )

            # use the weights
            if np.sum(convolved_weights == 0) > 0:
                print(
                    f"There are now {np.sum(convolved_weights == 0)} ill-defined weights, replacing the gains with zeros"
                )
                convolved_gains[np.where(convolved_weights == 0)] = 0
                convolved_weights[np.where(convolved_weights == 0)] = 1

            smoothed_gains[i, :, :] = convolved_gains / convolved_weights

        return smoothed_gains

    def _remove_unitairy_ambiguity(self, gains):
        """
        Set the phases of the reference station to zero to remove the unitairy ambiguity

        Parameters
        ----------
        gains : ndarray
            complex gains (frequency x station x patch)

        Returns
        -------
        ndarray
            gains after setting the reference station
        """
        amplitudes = np.abs(gains)
        phases = np.angle(gains)

        reference_phase = np.squeeze(
            phases[:, self.stations == self.reference_station, :], axis=1
        )
        corrected_phases = phases - reference_phase[:, np.newaxis, :]

        corrected_gains = amplitudes * np.exp(1j * corrected_phases)
        return corrected_gains

    def _select_baseline_data(self, data, station_number):
        """
        Select the baselines that correspond to a station in the data (so only the relevant baselines are returned).
        The data is transformed such that the selected station is always on the left-hand side (the gains are unconjugated)

        Parameters
        ----------
        data : ndarray
            data with baselines along the -1 axis
        station_number : int
            station to be selected

        Returns
        -------
        ndarray
            data with the relevant baselines
        """
        # select the data
        selected_data = data[..., self.baseline_mask[station_number, :]]

        # need to take the conjugate if the station is the second one
        selected_data[..., :station_number] = np.conj(
            selected_data[..., :station_number]
        )
        return selected_data

    def _DDEcal_timeslot(self, visibility, coherency):
        """
        Performs DDECal fully for a single time solution interval

        Parameters
        ----------
        visibility : ndarray
            visibility data (time x frequency x baseline)
        coherency : ndarray
            model data (patch x time x frequency x baseline)

        Returns
        -------
        dict
            "gains": gain solutions (frequency x station x patch),
            "residuals": visibility - model with gains per iteration,
            "loss": loss in the least squares problem (using previous iteration gains for the right-hand gain and the least-squares solution fo the left-hand gain),
            "n_iter": number of iterations,
        }
        """

        # Initialize gains
        gains = np.ones([self.n_spectral_sols, self.n_stations, self.n_patches]).astype(
            complex
        )
        new_gains = np.zeros_like(gains)

        # reshape data and model such that the station number is on the last axis, and always the first
        # station in the baseline (so the gain does not need to be conjugated)
        selected_visibilities = [
            self._select_baseline_data(visibility, i) for i in range(self.n_stations)
        ]
        selected_coherencies = [
            self._select_baseline_data(coherency, i) for i in range(self.n_stations)
        ]
        del visibility, coherency

        # compute the frequencies per solution interval and the weights that correspond to that slot
        weights = np.zeros_like(gains, dtype=float)
        for i in range(self.n_stations):
            for f in range(self.n_spectral_sols):
                selected_frequencies = range(
                    f * self.n_freqs_per_sol, (f + 1) * self.n_freqs_per_sol
                )
                weights[f, i, :] = self._reweight_function(
                    selected_coherencies[i][:, :, selected_frequencies, :]
                )

        # start the actual loop
        iteration = 0
        residuals = np.zeros(self.n_iterations)
        loss = np.zeros(self.n_iterations)
        while (
            iteration < self.n_iterations  # max its
            and self._change_rate(gains, new_gains)
            > self.tolerance  # check convergence
        ):
            for i, station in enumerate(self.stations):
                for f in range(self.n_spectral_sols):
                    # Give the visibilities for the frequencies in this slot and the baselines connected to this station
                    selected_frequencies = range(
                        f * self.n_freqs_per_sol, (f + 1) * self.n_freqs_per_sol
                    )

                    # Calculate new least-squares solution
                    new_gains[f, i, :], new_residual, new_loss = (
                        self._DDEcal_station_iteration(
                            gains=gains[f, :, :],
                            visibility=np.take(
                                selected_visibilities[i],
                                indices=selected_frequencies,
                                axis=1,
                            ),
                            coherency=np.take(
                                selected_coherencies[i],
                                indices=selected_frequencies,
                                axis=2,
                            ),
                            station_number=i,
                        )
                    )
                    residuals[iteration] += new_residual
                    loss[iteration] += new_loss

            # After all gains have received a new least-squares update direction, take a Gauss-Newton step
            gain_update = (
                1 - self.update_speed
            ) * gains + self.update_speed * new_gains

            # Regularise the new gains with the smoothness kernel
            gains = self._DDEcal_smooth_frequencies(
                gains=gain_update,
                weights=weights,
            )

            iteration += 1

        # Set the phases of the reference station to zero
        gains = self._remove_unitairy_ambiguity(gains)

        return {
            "gains": gains,
            "residuals": residuals[:iteration],
            "loss": loss[:iteration],
            "n_iter": iteration,
        }

    def _subtract_timeslot(self, visibility, coherency, gain):
        """
        Subtract the model from the visibilities for a single timeslot

        Parameters
        ----------
        visibility : ndarray
            visibility (data) (time x frequency x baseline)
        coherency : ndarray
            model (direction x time x frequency x baseline)
        gain : ndarray
            gains (frequency x station x direction)

        Returns
        -------
        ndarray
            residual visibilities (time x frequency x baseline)
        """
        subtractor = coherency.copy()  # Copy to apply the gains
        for i, station in enumerate(self.stations):
            for f in range(self.n_spectral_sols):
                # Select a solution interval in frequency
                selected_frequencies = range(
                    f * self.n_freqs_per_sol, (f + 1) * self.n_freqs_per_sol
                )

                # define whether the gains should be applied on the left or on the right (and thus conjugated)
                left_selection = self.baselines[:, 0] == station
                right_selection = self.baselines[:, 1] == station

                # applty the gains
                ndim = len(coherency[:, :, selected_frequencies, left_selection].shape)
                applied_gain = gain[f, i, :].reshape(-1, *([1] * (ndim - 1)))
                subtractor[:, :, selected_frequencies, left_selection] *= applied_gain
                subtractor[:, :, selected_frequencies, right_selection] *= np.conj(
                    applied_gain
                )

        # Subtract and return
        return visibility - np.sum(subtractor, axis=0)

    def _write_visibility(self, visibility, fname):
        """
        Write residuals to disk

        Parameters
        ----------
        visibility : ndarray
            residual visibilities (time x frequency x baseline)
        fname : str
            path + filename to write to in csv format
        """
        os.makedirs("/".join(fname.split("/")[:-1]), exist_ok=True)
        with open(fname, "w") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["time", "frequency", "baseline", "visibility"])
            dataset = [
                [
                    self.times[t],
                    self.frequencies[f],
                    "-".join(self.baselines[i]),
                    visibility[t, f, i],
                ]
                for t in range(self.n_times)
                for f in range(self.n_freqs)
                for i in range(self.n_baselines)
            ]
            writer.writerows(dataset)

    def run_DDEcal(
        self,
        visibility_file,
        skymodel,
        reuse_predict=False,
        reweight_mode=None,
        fname=None,
        calculate_residual=True,
        data_path=None,
    ):
        """
        Perform DDECal

        Parameters
        ----------
        visibility_file : str
            filename that contains the visibilities (path + name)
        skymodel : Skymodel object
            sources. Used for determining the calibration directions and possibly the predict, see sources.py
        reuse_predict : bool, optional
            If True, the coherency model is read from disk, if False it is computed first, by default False
        reweight_mode : str, optional
            Reweighting kernel ("abs", "squared" or None), by default None
            - abs: robust calibrator (smooth with the expected beam response)
            - squared: smooth with the square of the model times gain
            - None: smooth with the gain
        fname : str, optional
            Name of the output files. By default None (name is set to Calibration_<reweight_mode>_<smoothness_scale>).
        calculate_residual : bool, optional
            Whether to also calculate the residual visibilities, by default True
        data_path : str, optional
            basepath for the model and output, by default None (inferred from the visibility file)
        """
        # Set the path
        if data_path is None:
            self.data_path = "/".join(visibility_file.split("/")[:-1])
        else:
            self.data_path = data_path

        # Set the weighting kernel
        if reweight_mode == "abs":
            self._reweight_function = lambda coherency: np.nansum(
                np.abs(coherency), axis=(1, 2, 3)
            )
        elif reweight_mode == "squared":
            self._reweight_function = lambda coherency: np.nansum(
                np.abs(coherency) ** 2, axis=(1, 2, 3)
            )
        elif reweight_mode is None or reweight_mode == "none":
            self._reweight_function = lambda coherency: np.sum(
                ~np.isnan(coherency), axis=(1, 2, 3)
            )
        else:
            raise ValueError("Invalid reweight mode")

        # parse visibility
        visibilities = self._read_data(visibility_file, True)

        # Preflag on baseline length
        self._run_preflagger()
        if calculate_residual:
            flagged_visibilities = visibilities[self.flag_mask]
        visibilities[self.flag_mask] = np.nan

        # Obtain the model
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
        if calculate_residual:
            flagged_coherencies = coherency[:, self.flag_mask]
        coherency[:, self.flag_mask] = np.nan

        # select which rows of the visibility matrix will be active for each station (baseline mask)
        self.baseline_mask = np.ones([self.n_stations, self.n_baselines]).astype(bool)
        for i, station in enumerate(self.stations):
            self.baseline_mask[i, :] = (self.baselines[:, 0] == station) | (
                self.baselines[:, 1] == station
            )

        # Run DDECal for all timeslots in parallel
        results = Parallel(n_jobs=-1)(
            delayed(self._DDEcal_timeslot)(
                visibility=visibilities[t : t + self.n_times_per_sol, :, :],
                coherency=coherency[:, t : t + self.n_times_per_sol, :, :],
            )
            for t in np.arange(0, self.n_times, self.n_times_per_sol)
        )

        # Package the output and write
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
            fname = f"Calibration_{reweight_mode}_{int(self.smoothness_scale)}"

        os.makedirs(f"{self.data_path}/calibration_results/", exist_ok=True)
        with open(f"{self.data_path}/calibration_results/{fname}", "wb") as fp:
            pickle.dump(results, fp)
        with open(f"{self.data_path}/calibration_results/{fname}_metadata", "wb") as fp:
            pickle.dump(metadata, fp)

        # Calculate residual visibilties
        if calculate_residual:
            visibilities[self.flag_mask] = flagged_visibilities
            coherency[:, self.flag_mask] = flagged_coherencies

            print("calculating residual")
            visibility_residuals = Parallel(n_jobs=-1)(
                delayed(self._subtract_timeslot)(
                    visibility=visibilities[t : t + self.n_times_per_sol, :, :],
                    coherency=coherency[:, t : t + self.n_times_per_sol, :, :],
                    gain=results[slot_number]["gains"],
                )
                for slot_number, t in enumerate(
                    np.arange(0, self.n_times, self.n_times_per_sol)
                )
            )

            visibility_residuals = np.concatenate(visibility_residuals, axis=0)
            print(f"{self.data_path}/residuals/{fname}.csv")
            self._write_visibility(
                visibility_residuals, f"{self.data_path}/residuals/{fname}.csv"
            )

        return results

    def calculate_residuals(
        self, visibility_file, gain_path, skymodel, fname, data_path=None
    ):
        """
        Calculate residuals with known gains

        Parameters
        ----------
        visibility_file : str
            filename that contains the visibilities (path + name).
        gain_path : str
            path + filename of the gains
        skymodel : Skymodel object
            Used for determining the calibration directions, see sources.py
        fname : str
            output file name (including path)
        data_path : str, optional
            basepath for the model and gains, by default None (inferred from the visibility file)
        """
        # Set the path
        if data_path is None:
            self.data_path = "/".join(visibility_file.split("/")[:-1])
        else:
            self.data_path = data_path

        # Obtain data
        visibilities = self._read_data(visibility_file, True)

        # Obtain model
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

        # Obtain gains
        with open(gain_path, "rb") as fp:
            gains = pickle.load(fp)
        with open(f"{gain_path}_metadata", "rb") as fp:
            metadata = pickle.load(fp)

        # Select which calibration directions have gain solutions
        subtract_indices = [
            metadata["directions"].index(patch_name) for patch_name in patch_names
        ]

        # Apply the gains to the models and calculate residuals for all time intervals
        residuals = Parallel(n_jobs=-1)(
            delayed(self._subtract_timeslot)(
                visibility=visibilities[t : t + self.n_times_per_sol, :, :],
                coherency=coherency[:, t : t + self.n_times_per_sol, :, :],
                gain=gains[slot_number]["gains"].take(
                    indices=subtract_indices, axis=-1
                ),
            )
            for slot_number, t in enumerate(
                np.arange(0, self.n_times, self.n_times_per_sol)
            )
        )
        residuals = np.concatenate(residuals, axis=0)

        # Store the result
        self._write_visibility(residuals, fname)
