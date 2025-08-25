import numpy as np
from numpy.typing import NDArray
import os
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, List, Union
import datetime
import uproot
import glob
import h5py

import warnings
from libra_toolbox.neutron_detection.activation_foils.calibration import (
    CheckSource,
    ActivationFoil,
    na22,
    co60,
    ba133,
    mn54,
)
from libra_toolbox.neutron_detection.activation_foils.explicit import get_chain

from scipy.signal import find_peaks
from scipy.optimize import curve_fit


class Detector:
    """
    Represents a detector used in COMPASS measurements.

    This class stores detector events (time and energy pairs), channel number,
    and timing information.

    Attributes:
        events: Array of (time in ps, energy) pairs
        channel_nb: Channel number of the detector
        live_count_time: Active measurement time excluding dead time (in seconds)
        real_count_time: Total elapsed measurement time (in seconds)
        spectrum: Cached energy spectrum (accessed via property)
        bin_edges: Cached bin edges for the energy spectrum (accessed via property)
    """

    events: NDArray[Tuple[float, float]]  # type: ignore
    channel_nb: int
    live_count_time: Union[float, None]
    real_count_time: Union[float, None]
    _spectrum: Union[NDArray[np.float64], None] = None
    _bin_edges: Union[NDArray[np.float64], None] = None

    def __init__(self, channel_nb, nb_digitizer_bins=4096) -> None:
        """
        Initialize a Detector object.
        Args:
            channel_nb: channel number of the detector
            nb_digitizer_bins: number of digitizer bins for the detector.
        """
        self.channel_nb = channel_nb
        self.nb_digitizer_bins = nb_digitizer_bins
        self.events = np.empty((0, 2))  # Initialize as empty 2D array with 2 columns
        self.live_count_time = None
        self.real_count_time = None

    @property
    def spectrum(self) -> Union[NDArray[np.float64], None]:
        """Get the cached energy spectrum. Read-only property."""
        return getattr(self, "_spectrum", None)

    @property
    def bin_edges(self) -> Union[NDArray[np.float64], None]:
        """Get the cached bin edges for the energy spectrum. Read-only property."""
        return getattr(self, "_bin_edges", None)

    def get_energy_hist(
        self, bins: Union[None, NDArray[np.float64], int, str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the energy histogram of the detector events.
        Args:
            bins: bins for the histogram. If None, bins are automatically generated
                (one bin per energy channel). If int, it specifies the number of bins.
                If str, it specifies the binning method (e.g., 'auto', 'fd', etc.) see
                ``numpy.histogram_bin_edges`` for more details.
        Returns:
            Tuple of histogram values and bin edges
        """
        if self._spectrum is not None and self._bin_edges is not None:
            # If spectrum and bin edges are already calculated, return them
            return self._spectrum, self._bin_edges

        energy_values = self.events[:, 1].copy()
        time_values = self.events[:, 0].copy()

        # sort data based on timestamp
        inds = np.argsort(time_values)
        time_values = time_values[inds]
        energy_values = energy_values[inds]

        energy_values = np.nan_to_num(energy_values, nan=0)

        if bins is None:
            if self.nb_digitizer_bins == None:
                bins = np.arange(
                    int(np.nanmin(energy_values)), int(np.nanmax(energy_values)) + 1
                )
            else:
                bins = np.arange(self.nb_digitizer_bins + 1)

        return np.histogram(energy_values, bins=bins)

    def get_energy_hist_background_substract(
        self,
        background_detector: "Detector",
        bins: Union[NDArray[np.float64], None] = None,
        live_or_real: str = "live",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get the energy histogram of the detector events with background subtraction.

        Args:
            background_detector: _description_
            bins: _description_. Defaults to None.
            live_or_real: When doing the background sub decide whether the background
                histogram is scaled by live or real count time.
        """

        assert (
            self.channel_nb == background_detector.channel_nb
        ), f"Channel number mismatch: {self.channel_nb} != {background_detector.channel_nb}"

        raw_hist, raw_bin_edges = self.get_energy_hist(bins=bins)
        b_hist, _ = background_detector.get_energy_hist(bins=raw_bin_edges)

        if live_or_real == "live":
            # Scale background histogram by live count time
            b_hist = b_hist * (
                self.live_count_time / background_detector.live_count_time
            )
        elif live_or_real == "real":
            # Scale background histogram by real count time
            b_hist = b_hist * (
                self.real_count_time / background_detector.real_count_time
            )
        else:
            raise ValueError(
                f"Invalid live_or_real value: {live_or_real}. Use 'live' or 'real'."
            )

        hist_background_substracted = raw_hist - b_hist

        return hist_background_substracted, raw_bin_edges


class Measurement:
    """
    Represents a measurement session from a COMPASS detector system.

    The Measurement class encapsulates data from a complete measurement session,
    including timing information and detector events across multiple channels.
    It provides functionality to load and process measurement data from files
    generated by the COMPASS data acquisition system.

    Attributes:
        start_time: Start time of the measurement
        stop_time: End time of the measurement
        name: Identifier for this measurement
        detectors: List of ``Detector`` objects for each channel
    """

    start_time: Union[datetime.datetime, None]
    stop_time: Union[datetime.datetime, None]
    name: str
    detectors: List[Detector]

    def __init__(self, name: str) -> None:
        """
        Initialize a Measurement object.
        Args:
            name: name of the measurement
        """
        self.start_time = None
        self.stop_time = None
        self.name = name
        self.detectors = []

    @classmethod
    def from_directory(
        cls, source_dir: str, name: str, info_file_optional: bool = False
    ) -> "Measurement":
        """
        Create a Measurement object from a directory containing Compass data.
        Args:
            source_dir: directory containing Compass data
            name: name of the measurement
            info_file_optional: if True, the function will not raise an error
                if the run.info file is not found
        Returns:
            Measurement object
        """
        measurement_object = cls(name=name)

        # Get events
        time_values, energy_values = get_events(source_dir)

        # Get start and stop time
        try:
            start_time, stop_time = get_start_stop_time(source_dir)
            measurement_object.start_time = start_time
            measurement_object.stop_time = stop_time
        except FileNotFoundError as e:
            if info_file_optional:
                warnings.warn(
                    "run.info file not found. Assuming start and stop time are not needed."
                )
            else:
                raise FileNotFoundError(e)

        # Create detectors
        detectors = [Detector(channel_nb=nb) for nb in time_values.keys()]

        # Get live and real count times
        all_root_filenames = glob.glob(os.path.join(source_dir, "*.root"))
        if len(all_root_filenames) == 1:
            root_filename = all_root_filenames[0]
        else:
            root_filename = None
            print("No root file found, assuming all counts are live")

        for detector in detectors:
            detector.events = np.column_stack(
                (time_values[detector.channel_nb], energy_values[detector.channel_nb])
            )

            if root_filename:
                live_count_time, real_count_time = get_live_time_from_root(
                    root_filename, detector.channel_nb
                )
                detector.live_count_time = live_count_time
                detector.real_count_time = real_count_time
            else:
                real_count_time = (stop_time - start_time).total_seconds()
                # Assume first and last event correspond to start and stop time of live counts
                # and convert from picoseconds to seconds
                ps_to_seconds = 1e-12
                live_count_time = (
                    time_values[detector.channel_nb][-1]
                    - time_values[detector.channel_nb][0]
                ) * ps_to_seconds
                detector.live_count_time = live_count_time
                detector.real_count_time = real_count_time

        measurement_object.detectors = detectors

        return measurement_object

    def to_h5(self, filename: str, mode: str = "w", spectrum_only=False) -> None:
        """
        Save the measurement data to an HDF5 file.
        Args:
            filename: name of the output HDF5 file
            mode: file opening mode ('w' for write/overwrite, 'a' for append)
        """
        with h5py.File(filename, mode) as f:
            # Create a group for the measurement (or get existing one)
            if self.name in f:
                # If group already exists, we could either raise an error or overwrite
                # For now, let's overwrite the existing group
                del f[self.name]
            measurement_group = f.create_group(self.name)

            # Store start and stop time
            if self.start_time:
                measurement_group.attrs["start_time"] = self.start_time.isoformat()
            if self.stop_time:
                measurement_group.attrs["stop_time"] = self.stop_time.isoformat()

            # Store detectors
            for detector in self.detectors:
                detector_group = measurement_group.create_group(
                    f"detector_{detector.channel_nb}"
                )
                if spectrum_only:
                    hist, bin_edges = detector.get_energy_hist(bins=None)
                    detector_group.create_dataset("spectrum", data=hist)
                    detector_group.create_dataset("bin_edges", data=bin_edges)
                    detector_group.create_dataset("events", data=[])
                else:
                    detector_group.create_dataset("events", data=detector.events)

                detector_group.attrs["live_count_time"] = detector.live_count_time
                detector_group.attrs["real_count_time"] = detector.real_count_time

    @classmethod
    def from_h5(
        cls, filename: str, measurement_name: str = None
    ) -> Union["Measurement", List["Measurement"]]:
        """
        Load measurement data from an HDF5 file.
        Args:
            filename: name of the HDF5 file
            measurement_name: specific measurement name to load. If None, loads all measurements.
        Returns:
            Single Measurement object if measurement_name is specified,
            or list of Measurement objects if loading all measurements.
        """
        measurements = []

        with h5py.File(filename, "r") as f:
            # Get all measurement group names
            measurement_names = [
                name for name in f.keys() if isinstance(f[name], h5py.Group)
            ]

            if measurement_name is not None:
                if measurement_name not in measurement_names:
                    raise ValueError(
                        f"Measurement '{measurement_name}' not found in file. Available: {measurement_names}"
                    )
                measurement_names = [measurement_name]

            for name in measurement_names:
                measurement = cls(name=name)
                measurement_group = f[name]

                # Load start and stop time
                if "start_time" in measurement_group.attrs:
                    measurement.start_time = datetime.datetime.fromisoformat(
                        measurement_group.attrs["start_time"]
                    )
                if "stop_time" in measurement_group.attrs:
                    measurement.stop_time = datetime.datetime.fromisoformat(
                        measurement_group.attrs["stop_time"]
                    )

                # Load detectors
                detectors = []
                for detector_name in measurement_group.keys():
                    if detector_name.startswith("detector_"):
                        channel_nb = int(detector_name.replace("detector_", ""))
                        detector = Detector(channel_nb=channel_nb)

                        detector_group = measurement_group[detector_name]
                        detector.events = detector_group["events"][:]
                        detector.live_count_time = detector_group.attrs[
                            "live_count_time"
                        ]
                        detector.real_count_time = detector_group.attrs[
                            "real_count_time"
                        ]

                        if "spectrum" in detector_group:
                            detector._spectrum = detector_group["spectrum"][:]
                        if "bin_edges" in detector_group:
                            detector._bin_edges = detector_group["bin_edges"][:]

                        detectors.append(detector)

                measurement.detectors = detectors
                measurements.append(measurement)

        return measurements[0] if measurement_name is not None else measurements

    @classmethod
    def write_multiple_to_h5(
        cls, measurements: List["Measurement"], filename: str
    ) -> None:
        """
        Save multiple measurement objects to a single HDF5 file.
        Args:
            measurements: list of Measurement objects to save
            filename: name of the output HDF5 file
        """
        with h5py.File(filename, "w") as f:
            for measurement in measurements:
                # Create a group for each measurement
                measurement_group = f.create_group(measurement.name)

                # Store start and stop time
                if measurement.start_time:
                    measurement_group.attrs["start_time"] = (
                        measurement.start_time.isoformat()
                    )
                if measurement.stop_time:
                    measurement_group.attrs["stop_time"] = (
                        measurement.stop_time.isoformat()
                    )

                # Store detectors
                for detector in measurement.detectors:
                    detector_group = measurement_group.create_group(
                        f"detector_{detector.channel_nb}"
                    )
                    detector_group.create_dataset("events", data=detector.events)
                    detector_group.attrs["live_count_time"] = detector.live_count_time
                    detector_group.attrs["real_count_time"] = detector.real_count_time

    def get_detector(self, channel_nb: int) -> Detector:
        """
        Get the detector object for a given channel number.
        Args:
            channel_nb: channel number of the detector
        Returns:
            Detector object for the specified channel
        """
        for detector in self.detectors:
            if detector.channel_nb == channel_nb:
                return detector
        raise ValueError(f"Detector with channel number {channel_nb} not found.")


class CheckSourceMeasurement(Measurement):
    check_source: CheckSource

    def compute_detection_efficiency(
        self,
        background_measurement: Measurement,
        calibration_coeffs: np.ndarray,
        channel_nb: int,
        search_width: float = 800,
    ) -> Union[np.ndarray, float]:
        """
        Computes the detection efficiency of a check source given the
        check source data and the calibration coefficients.
        The detection efficiency is calculated using the formula:
        .. math:: \\eta = \\frac{N_{meas}}{N_{expec}}

        where :math:`N_{meas}` is the total number of counts measured under the energy peak
        and :math:`N_{expec}` is the total number of emitted gamma-rays from the check source.

        The expected number of counts :math:`N_{expec}` is calculated according to Equation 3
        in https://doi.org/10.2172/1524045.

        Args:
            background_measurement: background measurement
            calibration_coeffs: the calibration polynomial coefficients for the detector
            channel_nb: the channel number of the detector
            search_width: the search width for the peak fitting

        Returns:
            the detection efficiency
        """
        # find right background detector

        background_detector = background_measurement.get_detector(channel_nb)
        check_source_detector = self.get_detector(channel_nb)

        hist, bin_edges = check_source_detector.get_energy_hist_background_substract(
            background_detector, bins=None
        )

        calibrated_bin_edges = np.polyval(calibration_coeffs, bin_edges)

        nb_counts_measured = get_multipeak_area(
            hist,
            calibrated_bin_edges,
            self.check_source.nuclide.energy,
            search_width=search_width,
        )

        nb_counts_measured = np.array(nb_counts_measured)
        nb_counts_measured_err = np.sqrt(nb_counts_measured)

        # assert that all numbers in nb_counts_measured are > 0
        assert np.all(
            nb_counts_measured > 0
        ), f"Some counts measured are <= 0: {nb_counts_measured}"

        act_expec = self.check_source.get_expected_activity(self.start_time)
        gamma_rays_expected = act_expec * (
            np.array(self.check_source.nuclide.intensity)
        )
        decay_constant = np.log(2) / self.check_source.nuclide.half_life

        expected_nb_counts = gamma_rays_expected / decay_constant
        live_count_time_correction_factor = (
            check_source_detector.live_count_time
            / check_source_detector.real_count_time
        )
        decay_counting_correction_factor = 1 - np.exp(
            -decay_constant * check_source_detector.real_count_time
        )
        expected_nb_counts *= (
            live_count_time_correction_factor * decay_counting_correction_factor
        )

        detection_efficiency = nb_counts_measured / expected_nb_counts

        return detection_efficiency

    def get_peaks(self, hist: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the peak indices of the histogram

        Args:
            hist: a histogram
            kwargs: optional parameters for the peak finding algorithm
                see scipy.signal.find_peaks for more information

        Returns:
            the peak indices in ``hist``
        """

        # peak finding parameters
        start_index = 100
        prominence = 0.10 * np.max(hist[start_index:])
        height = 0.10 * np.max(hist[start_index:])
        width = [10, 150]
        distance = 30
        if self.check_source.nuclide == na22:
            start_index = 100
            height = 0.1 * np.max(hist[start_index:])
            prominence = 0.1 * np.max(hist[start_index:])
            width = [10, 150]
            distance = 30
        elif self.check_source.nuclide == co60:
            start_index = 400
            height = 0.60 * np.max(hist[start_index:])
            prominence = None
        elif self.check_source.nuclide == ba133:
            width = [10, 200]
        elif self.check_source.nuclide == mn54:
            height = 0.6 * np.max(hist[start_index:])

        # update the parameters if kwargs are provided
        if kwargs:
            start_index = kwargs.get("start_index", start_index)
            prominence = kwargs.get("prominence", prominence)
            height = kwargs.get("height", height)
            width = kwargs.get("width", width)
            distance = kwargs.get("distance", distance)

        # run the peak finding algorithm
        # NOTE: the start_index is used to ignore the low energy region
        peaks, peak_data = find_peaks(
            hist[start_index:],
            prominence=prominence,
            height=height,
            width=width,
            distance=distance,
        )
        peaks = np.array(peaks) + start_index

        return peaks


class SampleMeasurement(Measurement):
    foil: ActivationFoil

    def get_gamma_emitted(
        self,
        background_measurement: Measurement,
        efficiency_coeffs,
        calibration_coeffs,
        channel_nb: int,
        search_width: float = 800,
    ):
        # find right background detector

        background_detector = background_measurement.get_detector(channel_nb)
        check_source_detector = self.get_detector(channel_nb)

        hist, bin_edges = check_source_detector.get_energy_hist_background_substract(
            background_detector, bins=None
        )

        calibrated_bin_edges = np.polyval(calibration_coeffs, bin_edges)

        energy = self.foil.reaction.product.energy

        nb_counts_measured = get_multipeak_area(
            hist,
            calibrated_bin_edges,
            energy,
            search_width=search_width,
        )

        nb_counts_measured = np.array(nb_counts_measured)
        nb_counts_measured_err = np.sqrt(nb_counts_measured)

        detection_efficiency = np.polyval(efficiency_coeffs, energy)

        gamma_emmitted = nb_counts_measured / detection_efficiency
        gamma_emmitted_err = nb_counts_measured_err / detection_efficiency
        return gamma_emmitted, gamma_emmitted_err

    def get_neutron_flux(
        self,
        channel_nb: int,
        photon_counts: float,
        irradiations: list,
        time_generator_off: datetime.datetime,
        total_efficiency=1,
        branching_ratio=1,
    ):
        """calculates the neutron flux during the irradiation
        Based on Equation 1 from:
        Lee, Dongwon, et al. "Determination of the Deuterium-Tritium (D-T) Generator
        Neutron Flux using Multi-foil Neutron Activation Analysis Method." ,
        May. 2019. https://doi.org/10.2172/1524045

        Args:
            channel_nb: channel number of the detector
            irradiations: list of dictionaries with keys "t_on" and "t_off" for irradiations
            time_generator_off: time when the generator was turned off
            photon_counts: number of gamma rays measured
            total_efficiency: total efficiency of the detector
            branching_ratio: branching ratio of the reaction

        Returns:
            neutron flux in n/cm2/s
        """
        time_between_generator_off_and_start_of_counting = (
            self.start_time - time_generator_off
        ).total_seconds()

        detector = self.get_detector(channel_nb)

        f_time = (
            get_chain(irradiations, self.foil.reaction.product.decay_constant)
            * np.exp(
                -self.foil.reaction.product.decay_constant
                * time_between_generator_off_and_start_of_counting
            )
            * (
                1
                - np.exp(
                    -self.foil.reaction.product.decay_constant
                    * detector.real_count_time
                )
            )
            * (detector.live_count_time / detector.real_count_time)
            / self.foil.reaction.product.decay_constant
        )

        # Correction factor of gamma-ray self-attenuation in the foil
        if self.foil.thickness is None:
            f_self = 1
        else:
            f_self = (
                1
                - np.exp(
                    -self.foil.mass_attenuation_coefficient
                    * self.foil.density
                    * self.foil.thickness
                )
            ) / (
                self.foil.mass_attenuation_coefficient
                * self.foil.density
                * self.foil.thickness
            )

        # Spectroscopic Factor to account for the branching ratio and the
        # total detection efficiency
        f_spec = total_efficiency * branching_ratio

        number_of_decays_measured = photon_counts / f_spec

        flux = (
            number_of_decays_measured
            / self.foil.nb_atoms
            / self.foil.reaction.cross_section
        )

        flux /= f_time * f_self

        return flux

    def get_neutron_rate(
        self,
        channel_nb: int,
        photon_counts: float,
        irradiations: list,
        distance: float,
        time_generator_off: datetime.datetime,
        total_efficiency=1,
        branching_ratio=1,
    ) -> float:
        """
        Calculates the neutron rate during the irradiation.
        It assumes that the neutron flux is isotropic.

        Based on Equation 1 from:
        Lee, Dongwon, et al. "Determination of the Deuterium-Tritium (D-T) Generator
        Neutron Flux using Multi-foil Neutron Activation Analysis Method." ,
        May. 2019. https://doi.org/10.2172/1524045

        Args:
            channel_nb: channel number of the detector
            irradiations: list of dictionaries with keys "t_on" and "t_off" for irradiations
            time_generator_off: time when the generator was turned off
            photon_counts: number of gamma rays measured
            total_efficiency: total efficiency of the detector
            branching_ratio: branching ratio of the reaction

        Returns:
            neutron rate in n/s
        """

        flux = self.get_neutron_flux(
            channel_nb=channel_nb,
            photon_counts=photon_counts,
            irradiations=irradiations,
            time_generator_off=time_generator_off,
            total_efficiency=total_efficiency,
            branching_ratio=branching_ratio,
        )
        # convert n/cm2/s to n/s
        area_of_sphere = 4 * np.pi * distance**2

        flux *= area_of_sphere

        return flux


def get_calibration_data(
    check_source_measurements: List[CheckSourceMeasurement],
    background_measurement: Measurement,
    channel_nb: int,
):
    background_detector = [
        detector
        for detector in background_measurement.detectors
        if detector.channel_nb == channel_nb
    ][0]

    calibration_energies = []
    calibration_channels = []

    for measurement in check_source_measurements:
        for detector in measurement.detectors:
            if detector.channel_nb != channel_nb:
                continue

            hist, bin_edges = detector.get_energy_hist_background_substract(
                background_detector, bins=None
            )
            peaks_ind = measurement.get_peaks(hist)
            peaks = bin_edges[peaks_ind]

            if len(peaks) != len(measurement.check_source.nuclide.energy):
                raise ValueError(
                    f"SciPy find_peaks() found {len(peaks)} photon peaks, while {len(measurement.check_source.nuclide.energy)} were expected"
                )
            calibration_channels += list(peaks)
            calibration_energies += measurement.check_source.nuclide.energy

    inds = np.argsort(calibration_channels)
    calibration_channels = np.array(calibration_channels)[inds]
    calibration_energies = np.array(calibration_energies)[inds]

    return calibration_channels, calibration_energies


def gauss(x, b, m, *args):
    """Creates a multipeak gaussian with a linear addition of the form:
    m * x + b + Sum_i (A_i * exp(-(x - x_i)**2) / (2 * sigma_i**2)"""

    out = m * x + b
    if np.mod(len(args), 3) == 0:
        for i in range(int(len(args) / 3)):
            out += args[i * 3 + 0] * np.exp(
                -((x - args[i * 3 + 1]) ** 2) / (2 * args[i * 3 + 2] ** 2)
            )
    else:
        raise ValueError("Incorrect number of gaussian arguments given.")
    return out


def fit_peak_gauss(hist, xvals, peak_ergs, search_width=600, threshold_overlap=200):

    if len(peak_ergs) > 1:
        if np.max(peak_ergs) - np.min(peak_ergs) > threshold_overlap:
            raise ValueError(
                f"Peak energies {peak_ergs} are too far away from each to be fitted together."
            )

    search_start = np.argmin(
        np.abs((peak_ergs[0] - search_width / (2 * len(peak_ergs))) - xvals)
    )
    search_end = np.argmin(
        np.abs((peak_ergs[-1] + search_width / (2 * len(peak_ergs))) - xvals)
    )

    slope_guess = (hist[search_end] - hist[search_start]) / (
        xvals[search_end] - xvals[search_start]
    )

    guess_parameters = [0, slope_guess]

    for i in range(len(peak_ergs)):
        peak_ind = np.argmin(np.abs((peak_ergs[i]) - xvals))
        guess_parameters += [
            hist[peak_ind],
            peak_ergs[i],
            search_width / (3 * len(peak_ergs)),
        ]

    parameters, covariance = curve_fit(
        gauss,
        xvals[search_start:search_end],
        hist[search_start:search_end],
        p0=guess_parameters,
    )

    return parameters, covariance


def get_multipeak_area(
    hist, bins, peak_ergs, search_width=600, threshold_overlap=200
) -> List[float]:

    if len(peak_ergs) > 1:
        if np.max(peak_ergs) - np.min(peak_ergs) > threshold_overlap:
            areas = []
            for peak in peak_ergs:
                area = get_multipeak_area(
                    hist,
                    bins,
                    [peak],
                    search_width=search_width,
                    threshold_overlap=threshold_overlap,
                )
                areas += area
            return areas

    # get midpoints of every bin
    xvals = np.diff(bins) / 2 + bins[:-1]

    parameters, covariance = fit_peak_gauss(
        hist, xvals, peak_ergs, search_width=search_width
    )

    areas = []
    peak_starts = []
    peak_ends = []
    all_peak_params = []
    # peak_amplitudes = []
    for i in range(len(peak_ergs)):
        # peak_amplitudes += [parameters[2 + 3 * i]]
        mean = parameters[2 + 3 * i + 1]
        sigma = np.abs(parameters[2 + 3 * i + 2])
        peak_start = np.argmin(np.abs((mean - 3 * sigma) - xvals))
        peak_end = np.argmin(np.abs((mean + 3 * sigma) - xvals))

        peak_starts += [peak_start]
        peak_ends += [peak_end]

        # Use unimodal gaussian to estimate counts from just one peak
        peak_params = [parameters[0], parameters[1], parameters[2 + 3 * i], mean, sigma]
        all_peak_params += [peak_params]
        gross_area = np.trapz(
            gauss(xvals[peak_start:peak_end], *peak_params),
            x=xvals[peak_start:peak_end],
        )

        # Cut off trapezoidal area due to compton scattering and noise
        trap_cutoff_area = np.trapz(
            parameters[0] + parameters[1] * xvals[peak_start:peak_end],
            x=xvals[peak_start:peak_end],
        )
        area = gross_area - trap_cutoff_area
        areas += [area]

    return areas


def get_channel(filename):
    """
    Extract the channel number from a given filename string.

    Parameters
    ----------
    filename : str
        The input filename string containing the channel information.
        Should look something like : "Data_CH<channel_number>@V...CSV"

    Returns
    -------
    int
        The extracted channel number.

    Example
    -------
    >>> get_channel("Data_CH4@V1725_292_Background_250322.CSV")
    4
    """
    return int(filename.split("@")[0][7:])


def sort_compass_files(directory: str) -> dict:
    """Gets Compass csv data filenames
    and sorts them according to channel and ending number.
    The filenames need to be sorted by ending number because only
    the first csv file for each channel contains a header.

    Example of sorted filenames in directory:
        1st file: Data_CH4@...22.CSV
        2nd file: Data_CH4@...22_1.CSV
        3rd file: Data_CH4@...22_2.CSV"""

    filenames = os.listdir(directory)
    data_filenames = {}
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            ch = get_channel(filename)
            # initialize filenames for each channel
            if ch not in data_filenames.keys():
                data_filenames[ch] = []

            data_filenames[ch].append(filename)
    # Sort filenames by number at end
    for ch in data_filenames.keys():
        data_filenames[ch] = np.sort(data_filenames[ch])

    return data_filenames


def get_events(directory: str) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    From a directory with unprocessed Compass data CSV files,
    this returns dictionaries of detector pulse times and energies
    with digitizer channels as the keys to the dictionaries.

    This function is also built to be able to read-in problematic
    Compass CSV files that have been incorrectly post-processed to
    reduce waveform data.

    Args:
        directory: directory containing CSV files with Compass data

    Returns:
        time values and energy values for each channel
    """

    time_values = {}
    energy_values = {}

    data_filenames = sort_compass_files(directory)

    for ch in data_filenames.keys():
        # Initialize time_values and energy_values for each channel
        time_values[ch] = np.empty(0)
        energy_values[ch] = np.empty(0)
        for i, filename in enumerate(data_filenames[ch]):
            print(f'Processing File {i}')

            csv_file_path = os.path.join(directory, filename)

            # only the first file has a header
            if i == 0:
                # determine the column names
                # 
                # Typically, setting the header argument to 1
                # would normally work, but on some CoMPASS csv
                # files, specifically those with waveform data,
                # the column header has far fewer entries
                # than the number of columns in the csv file.
                # This is due to the "SAMPLES" column, which 
                # contains the waveform data actually being made
                # up of the 7th-nth column of an n column csv file.
                #
                # So to mitigate this, we will read in the header
                # manually and determine which column of 
                # the dataset to read in. 
                first_row_df = pd.read_csv(csv_file_path,
                                           delimiter=";",
                                           header=None,
                                           nrows=1)
                column_names = first_row_df.to_numpy()[0]
                # Determine which column applies to time and energy
                time_col = np.where(column_names=="TIMETAG")[0][0]
                energy_col = np.where(column_names=="ENERGY")[0][0]
                # First csv file has header, so skip it
                # because we already read it in
                skiprows=1
            else:
                # For subsequent csv files, don't skip any rows
                # as there won't be any header
                skiprows=0


            df = pd.read_csv(csv_file_path, 
                             delimiter=";", 
                             header=None,
                             skiprows=skiprows)

            time_data = df[time_col].to_numpy()
            energy_data = df[energy_col].to_numpy()

            # Extract and append the energy data to the list
            time_values[ch] = np.concatenate([time_values[ch], time_data])
            energy_values[ch] = np.concatenate([energy_values[ch], energy_data])

    return time_values, energy_values


def get_start_stop_time(directory: str) -> Tuple[datetime.datetime, datetime.datetime]:
    """Obtains count start and stop time from the run.info file."""

    info_file = Path(directory).parent / "run.info"
    if info_file.exists():
        time_format = "%Y/%m/%d %H:%M:%S.%f%z"
        with open(info_file, "r") as file:
            lines = file.readlines()
    else:
        raise FileNotFoundError(
            f"Could not find run.info file in parent directory {Path(directory).parent}"
        )

    start_time, stop_time = None, None
    for line in lines:
        if "time.start=" in line:
            # get start time string while cutting off '\n' newline
            time_string = line.split("=")[1].replace("\n", "")
            start_time = datetime.datetime.strptime(time_string, time_format)
        elif "time.stop=" in line:
            # get stop time string while cutting off '\n' newline
            time_string = line.split("=")[1].replace("\n", "")
            stop_time = datetime.datetime.strptime(time_string, time_format)

    if None in (start_time, stop_time):
        raise ValueError(f"Could not find time.start or time.stop in file {info_file}.")
    else:
        return start_time, stop_time


def get_live_time_from_root(root_filename: str, channel: int) -> Tuple[float, float]:
    """
    Gets live and real count time from Compass root file.
    Live time is defined as the difference between the actual time that
    a count is occurring and the "dead time," in which the output of detector
    pulses is saturated such that additional signals cannot be processed."""

    with uproot.open(root_filename) as root_file:
        live_count_time = root_file[f"LiveTime_{channel}"].members["fMilliSec"] / 1000
        real_count_time = root_file[f"RealTime_{channel}"].members["fMilliSec"] / 1000
    return live_count_time, real_count_time
