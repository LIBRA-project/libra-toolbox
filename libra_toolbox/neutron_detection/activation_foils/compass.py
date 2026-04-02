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
import xml.etree.ElementTree as ET
import re
from zoneinfo import ZoneInfo

import warnings
from libra_toolbox.neutron_detection.activation_foils.calibration import (
    CheckSource,
    ActivationFoil,
    na22,
    co60,
    ba133,
    mn54,
    cs137
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
    detector_type: str = "NaI"  # Default detector type, can be 'NaI' or 'HPGe'

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
        # First check if root files are present in the source directory and get live and real count times from there
        all_root_filenames = glob.glob(os.path.join(source_dir, "*.root"))
        if len(all_root_filenames) == 1:
            root_filename = all_root_filenames[0]
        else:
            root_filename = None
            
        # if root file is not present, check for *_info.txt files and get live and real count times from there
        if root_filename is None:
            info_txt_filename = Path(source_dir).parent / f"{Path(source_dir).parent.stem}_info.txt"
            if not os.path.isfile(info_txt_filename):
                # if the *_info.txt file is not present, set to None, 
                # which will assume that the live count time is the time between the first and last event, 
                # and the real count time is the time between the start and stop time (if they are available)
                info_txt_filename = None

        
        # Get energy channel bins for each detector from the settings.xml file if it exists, otherwise assume 4096 bins
        # search for settings.xml file in source_dir
        settings_file = Path(source_dir).parent / "settings.xml"
        if os.path.isfile(settings_file):
            energy_bins = get_spectrum_nbins(settings_file)
            print(f"Found settings.xml file. Using energy bins from file: {energy_bins}")
        else:
            print("No settings.xml file found. Assuming 4096 energy bins.")
            print(os.listdir(Path(source_dir).parent))
            energy_bins = None
        if not energy_bins:
            energy_bins = 4096

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
                if info_txt_filename:
                    live_count_time, real_count_time = get_live_time_from_info_txt(
                        info_txt_filename, detector.channel_nb
                    )
                    detector.live_count_time = live_count_time
                    detector.real_count_time = real_count_time
                else:
                    real_count_time = (stop_time - start_time).total_seconds()
                    print("Assuming real count time is the time between start and stop time: ", real_count_time)
                    # Assume first and last event correspond to start and stop time of live counts
                    # and convert from picoseconds to seconds
                    ps_to_seconds = 1e-12
                    live_count_time = (
                        time_values[detector.channel_nb][-1]
                        - time_values[detector.channel_nb][0]
                    ) * ps_to_seconds
                    print("Assuming live count time is the time between first and last event: ", live_count_time)
                    detector.live_count_time = live_count_time
                    detector.real_count_time = real_count_time
            detector.nb_digitizer_bins = energy_bins

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
                detector_group.attrs["nb_digitizer_bins"] = detector.nb_digitizer_bins

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
                        
                        detector.nb_digitizer_bins = detector_group.attrs.get("nb_digitizer_bins", 4096)

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
    _uncalibrated_measured_energies: Dict[int, List[float]] = None
    """ 
    check_source: CheckSource object containing the information of the check source used in the measurement.
    _uncalibrated_measured_energies: Dictionary to store the uncalibrated measured energies of the check source photopeaks
                                    for each channel number. The keys are the channel numbers and the values are lists of 
                                    uncalibrated measured energies of the photopeaks for that detector channel.
                                    This is used to store the energy channels associated with each photopeak
                                    for later use in calculating the area under each peak and the detection efficiency
                                    in compute_detection_efficiency(). This is necessary because the energy channels 
                                    of the photopeaks may not be the same as the actual energies of the photopeaks 
                                    due to the calibration process.
    """

    def get_calibrated_measured_energies(self, channel_nb: int, calibration_coeffs: np.ndarray) -> List[float]:
        """
        Returns the calibrated measured energies of the check source for a given channel number and calibration coefficients.
        The reason this is needed is that due to imperfect calibration, the measured energy of the photopeaks may
        not be the same as the actual energies of the photopeaks. This function applies the calibration coefficients 
        to the uncalibrated measured energies to get the calibrated measured energies, so that when calculating the detector
        efficiency with compute_detection_efficiency(), the correct energies are used to find the area under each peak and 
        the expected number of counts.

        Args:
            channel_nb: channel number of the detector
            calibration_coeffs: polynomial coefficients for energy calibration
        Returns:
            List of calibrated measured photopeak energies in keV.
            Should be the same length as self.check_source.nuclide.energy
        """
        if self._uncalibrated_measured_energies is None:
            return None
        else:
            uncalibrated = np.array(
                self._uncalibrated_measured_energies.get(channel_nb, []),
                dtype=float
            )
            return np.polyval(calibration_coeffs, uncalibrated)

    def compute_detection_efficiency(
        self,
        background_measurement: Measurement,
        calibration_coeffs: np.ndarray,
        channel_nb: int,
        search_width: float = 800,
        compute_error: bool = False,
        threshold_overlap: float = 200,
        summing_method: str = 'sum_gaussian',
        ax=None
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
            threshold_overlap: the threshold width for considering two peaks as overlapping
            summing_method: method to sum counts under the peak, either 'sum_gaussian' or 'sum_histogram'
                            with 'sum_gaussian' fitting a Gaussian to the peak and integrating it, 
                            and with 'sum_histogram' summing the histogram counts under the peak.
                            'sum_histogram' SHOULD NOT BE USED for OVERLAPPING peaks as it will
                            overestimate the counts.

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

        peak_energies = self.get_calibrated_measured_energies(channel_nb, calibration_coeffs)
        if peak_energies is None:
            peak_energies = self.check_source.nuclide.energy

        nb_counts_measured = get_multipeak_area(
            hist,
            calibrated_bin_edges,
            peak_energies,
            search_width=search_width,
            threshold_overlap=threshold_overlap,
            summing_method=summing_method,
            ax=ax
        )

        nb_counts_measured = np.array(nb_counts_measured)
        print("Measured counts under peaks for ", self.check_source.nuclide.name, ": ", nb_counts_measured)

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

        if compute_error:
            nb_counts_measured_err = np.sqrt(nb_counts_measured)

            act_expec_err = self.check_source.get_expected_activity_error(self.start_time)
            gamma_rays_expected_err = act_expec_err * (
                np.array(self.check_source.nuclide.intensity)
            )
            expected_nb_counts_err = gamma_rays_expected_err / decay_constant
            expected_nb_counts_err *= (
                live_count_time_correction_factor * decay_counting_correction_factor
            )
            
            detection_efficiency_err = detection_efficiency * np.sqrt(
                (nb_counts_measured_err / nb_counts_measured) ** 2 +
                (expected_nb_counts_err / expected_nb_counts) ** 2
            )
            return detection_efficiency, detection_efficiency_err
        else:
            return detection_efficiency

    def get_peaks(self, hist: np.ndarray, **kwargs) -> np.ndarray:
        """Returns the peak indices of the histogram

        Args:
            hist: a histogram
            kwargs: optional parameters for the peak finding algorithm
                see scipy.signal.find_peaks for more information like:
                    start_index: the index to start the peak finding from, to ignore the low energy region

                    relative_prominence: fraction of maximum count in the histogram to set the prominence parameter for peak finding

                    relative_height: fraction of maximum count in the histogram to set the height parameter for peak finding

                    width: the width parameter for peak finding
                    
                    distance: the distance parameter for peak finding

        Returns:
            the peak indices in ``hist``
        """

        # get total number of channels for scaling
        total_channels = self.detectors[0].nb_digitizer_bins
        channel_multiplier = int(total_channels / 4096)  # Assuming 4096 channels is the base case
        if self.detector_type.lower() == 'nai':
            # peak finding parameters
            start_index = int(100 * channel_multiplier)
            relative_prominence = 0.10
            relative_height = 0.10
            width = [int(10 * channel_multiplier), int(150 * channel_multiplier)]
            distance = int(30 * channel_multiplier)
            if self.check_source.nuclide == na22:
                start_index = int(100 * channel_multiplier)
                relative_height = 0.1
                relative_prominence = 0.1
                width = [int(10 * channel_multiplier), int(150 * channel_multiplier)]
                distance = int(30 * channel_multiplier)
            elif self.check_source.nuclide == co60:
                start_index = int(400 * channel_multiplier)
                relative_height = 0.60
                relative_prominence = None
            elif self.check_source.nuclide == ba133:
                start_index = int(150 * channel_multiplier)
                relative_height = 0.10
                relative_prominence = 0.10
            elif self.check_source.nuclide == mn54:
                relative_height = 0.6 
        elif self.detector_type.lower() == 'hpge':
            # peak finding parameters for HPGe detectors
            start_index = int(10 * channel_multiplier)
            relative_prominence = 0.50
            relative_height = 0.50
            width = [int(2 * channel_multiplier), int(50 * channel_multiplier)]
            distance = int(100 * channel_multiplier)
            if self.check_source.nuclide == na22:
                start_index = int(100 * channel_multiplier)
                relative_height = 0.4
                relative_prominence = 0.4
                distance = int(100 * channel_multiplier)
            elif self.check_source.nuclide == co60:
                relative_height = 0.5
                relative_prominence = 0.5
            elif self.check_source.nuclide == ba133:
                start_index = int(150 * channel_multiplier)
                relative_height = 0.10
                relative_prominence = 0.10
                distance = int(10 * channel_multiplier)
            elif self.check_source.nuclide == mn54:
                start_index = int(400 * channel_multiplier)
                relative_height = 0.7
                relative_prominence = 0.7
                distance = int(100 * channel_multiplier)
        elif self.detector_type.lower() == 'labr':
            # peak finding parameters for LaBr3 detectors
            start_index = int(250 * channel_multiplier)
            relative_prominence = 0.30
            relative_height = 0.30
            width = [int(1 * channel_multiplier), int(40 * channel_multiplier)]
            distance = int(10 * channel_multiplier)
            if self.check_source.nuclide == na22:
                start_index = int(400 * channel_multiplier)
                relative_height = 0.1
                relative_prominence = 0.1
                width = None
            elif self.check_source.nuclide == ba133:
                start_index = int(300 * channel_multiplier)
                relative_prominence = 0.05
                relative_height = 0.1
            elif self.check_source.nuclide == co60:
                start_index = int(500 * channel_multiplier)
                relative_prominence = 0.5
                relative_height = 0.5
            elif self.check_source.nuclide == cs137:
                start_index = int(400 * channel_multiplier)
                relative_prominence = 0.5
                relative_height = 0.5
                width = [int(1 * channel_multiplier), int(50 * channel_multiplier)]
        else:
            raise ValueError(
                f"Unknown detector type: {self.detector_type}. Supported types are 'NaI' and 'HPGe'."
            )

        # update the parameters if kwargs are provided
        if kwargs:
            print("kwargs provided, updating peak finding parameters with provided values")
            start_index = kwargs.get("start_index", start_index)
            relative_prominence = kwargs.get("relative_prominence", relative_prominence)
            relative_height = kwargs.get("relative_height", relative_height)
            width = kwargs.get("width", width)
            distance = kwargs.get("distance", distance)

        prominence = relative_prominence * np.max(hist) if relative_prominence is not None else None
        height = relative_height * np.max(hist) if relative_height is not None else None
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

        # special case for Mn-54, only keep the first high count energy peak
        if self.check_source.nuclide == mn54 and len(peaks) > 1:
            peaks = np.array([peaks[0]])

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
        detection_efficiency: float = None,
        detection_efficiency_err: float = 0.0,
        summing_method: str = 'sum_gaussian',
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
            summing_method=summing_method,
        )

        nb_counts_measured = np.array(nb_counts_measured)
        nb_counts_measured_err = np.sqrt(nb_counts_measured)

        if detection_efficiency is None:
            detection_efficiency = np.polyval(efficiency_coeffs, energy)

        gamma_emmitted = nb_counts_measured / detection_efficiency
        gamma_emmitted_err = gamma_emmitted * np.sqrt(
            (nb_counts_measured_err / nb_counts_measured) ** 2
            + (detection_efficiency_err / detection_efficiency) ** 2
        )
        
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

        print(f"number_of_decays_measured: {number_of_decays_measured}")
        print("nnumber of atoms in foil: ", self.foil.nb_atoms)
        print("cross section: ", self.foil.reaction.cross_section)
        flux = (
            number_of_decays_measured
            / self.foil.nb_atoms
            / self.foil.reaction.cross_section
        )
        # If the cross section is zero or negative, set the flux to zero to avoid division by zero or negative flux values
        if np.any(self.foil.reaction.cross_section <= 0):
            zero_indices = np.where(self.foil.reaction.cross_section <= 0)[0]
            flux[zero_indices] = 0
        print(f"flux before time and self-attenuation correction: {flux}")

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
    peak_kwargs: dict = None,
):
    """
    Finds the radionuclide peaks from the check source measurements and returns
    a list of the energy channels and a list of the actual energies associated 
    with those peaks.
    
    check_source_measurements: list of CheckSourceMeasurement objects
    background_measurement: Measurement object for the background measurement
    channel_nb: channel number of the detector to use for calibration
    peak_kwargs: optional dictionary of keyword arguments to pass to the function 
        get_peaks() for each check source measurement, with the check source nuclide 
        name as key.
        Example: peak_kwargs = {
            'Na-22': {'start_index': 100, 'height': 0.1},
            'Co-60': {'start_index': 400, 'height': 0.6},
        }
        "The height and prominence parameters in the get_peaks() function are defined 
        as a factor of the maximum count in the histogram, starting from the start_index.
        So for example, if the histogram has a maximum count of 1000 starting from the start_index, 
        and the height parameter is set to 0.1, then the height threshold for peak finding will be 100 counts.
        This allows the peak finding parameters to be scaled according to the actual counts 
        in the histogram for each check source measurement, which can be useful if the check sources 
        have different activities and therefore different count rates.
    """
    background_detector = [
        detector
        for detector in background_measurement.detectors
        if detector.channel_nb == channel_nb
    ][0]

    calibration_energies = []
    calibration_channels = []
    found_a_nuclide = False
    for measurement in check_source_measurements:
        for detector in measurement.detectors:
            if detector.channel_nb != channel_nb:
                continue

            hist, bin_edges = detector.get_energy_hist_background_substract(
                background_detector, bins=None
            )
            kwargs = {}
            if peak_kwargs is not None:
                if measurement.check_source.nuclide.name in peak_kwargs.keys():
                    kwargs = peak_kwargs[measurement.check_source.nuclide.name]
                    found_a_nuclide = True

            peaks_ind = measurement.get_peaks(hist, **kwargs)
            peaks = bin_edges[peaks_ind]

            if len(peaks) != len(measurement.check_source.nuclide.energy):
                raise ValueError(
                    f"SciPy find_peaks() found {len(peaks)} photon peaks, while {len(measurement.check_source.nuclide.energy)} were expected",
                    f" peaks found: {peaks} for {measurement.check_source.nuclide.name}",
                )
            calibration_channels += list(peaks)
            calibration_energies += measurement.check_source.nuclide.energy
            # Store the uncalibrated measured energies in the measurement object for later use
            if measurement._uncalibrated_measured_energies is None:
                measurement._uncalibrated_measured_energies = {}
            measurement._uncalibrated_measured_energies[channel_nb] = list(peaks)

    if not found_a_nuclide and peak_kwargs is not None:
        warnings.warn(
            "No check source nuclide found in the provided peak_kwargs. The default peak finding parameters will be used for all check sources."
        )
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


def fit_peak_gauss(hist, xvals, peak_ergs, 
                   search_width=600, 
                   threshold_overlap=200,
                   ax=None):

    if len(peak_ergs) > 1:
        if np.max(peak_ergs) - np.min(peak_ergs) > threshold_overlap:
            raise ValueError(
                f"Peak energies {peak_ergs} are too far away from each to be fitted together."
            )

    search_start = np.argmin(
        np.abs((peak_ergs[0] - search_width / ( len(peak_ergs))) - xvals)
    )
    search_end = np.argmin(
        np.abs((peak_ergs[-1] + search_width / (len(peak_ergs))) - xvals)
    )

    slope_guess = (hist[search_end] - hist[search_start]) / (
        xvals[search_end] - xvals[search_start]
    )

    # guess_parameters = [0, slope_guess]
    guess_parameters = [0, 0]

    for i in range(len(peak_ergs)):
        peak_ind = np.argmin(np.abs((peak_ergs[i]) - xvals))
        guess_parameters += [
            hist[peak_ind],
            peak_ergs[i],
            search_width / (3 * len(peak_ergs)),
        ]

    if ax:
        print("Plotting initial guess...")
        ax.plot(xvals[search_start:search_end], gauss(xvals[search_start:search_end], *guess_parameters), 
                '--',
                label='Initial Guess')

    parameters, covariance = curve_fit(
        gauss,
        xvals[search_start:search_end],
        hist[search_start:search_end],
        p0=guess_parameters,
    )

    if ax:
        print("Plotting fitted curve...")
        ax.plot(xvals[search_start:search_end], gauss(xvals[search_start:search_end], *parameters), 
                label='Fitted Curve')
        ax.legend()



    return parameters, covariance


def get_multipeak_area(
    hist, 
    bins, 
    peak_ergs, 
    search_width=600, 
    threshold_overlap=200,
    summing_method='sum_gaussian',
    ax=None,
) -> List[float]:
    
    print(peak_ergs)

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
                    summing_method=summing_method,
                    ax=ax
                )
                areas += area
            return areas
    

    # get midpoints of every binß
    xvals = np.diff(bins) / 2 + bins[:-1]


    parameters, covariance = fit_peak_gauss(
        hist, xvals, peak_ergs, search_width=search_width,
        ax=ax,
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

        if summing_method == 'sum_gaussian':
            gross_area = np.trapezoid(
                gauss(xvals[peak_start:peak_end], *peak_params),
                x=xvals[peak_start:peak_end],
            )
        elif summing_method == 'sum_histogram':
            gross_area = np.sum(
                (
                    hist[peak_start:peak_end]
                    * np.diff(bins[peak_start:peak_end+1])
            ))
        # Cut off trapezoidal area due to compton scattering and noise
        trap_cutoff_area = np.trapezoid(
            parameters[0] + parameters[1] * xvals[peak_start:peak_end],
            x=xvals[peak_start:peak_end],
        )
        area = gross_area - trap_cutoff_area

        areas += [area]


    return areas


def plot_histogram_with_peak_fit(
    detector: Detector,
    calibration_coeffs: np.ndarray,
    peak_energies: List[float],
    background_detector: Detector = None,
    search_width: float = 600,
    threshold_overlap: float = 200,
    ax=None,
    plot_initial_guess: bool = False,
    plot_peak_area: bool = True,
    hist_kwargs: dict = None,
    fit_kwargs: dict = None,
    fill_kwargs: dict = None,
    plot_title: str = None,
):
    """
    Plot the energy histogram of a detector with Gaussian peak fits.
    
    Args:
        detector: Detector object containing the measurement data
        calibration_coeffs: Polynomial coefficients for energy calibration
        peak_energies: List of peak energies (in keV) to fit
        background_detector: Optional background Detector for subtraction
        search_width: Width around peaks to search for fitting
        threshold_overlap: Threshold for considering peaks as overlapping
        ax: Matplotlib axes object. If None, a new figure is created.
        plot_initial_guess: If True, also plot the initial guess for the fit
        plot_peak_area: If True, fill the area under the peak down to the baseline
        hist_kwargs: Additional kwargs for histogram plotting (e.g., color, alpha)
        fit_kwargs: Additional kwargs for fit curve plotting (e.g., color, linewidth)
        fill_kwargs: Additional kwargs for area fill (e.g., color, alpha)
        
    Returns:
        Tuple of (fig, ax, fit_parameters) where fit_parameters is a list of
        fitted parameters for each peak group
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()
    
    # Get histogram data
    if background_detector is not None:
        hist, bin_edges = detector.get_energy_hist_background_substract(
            background_detector, bins=None
        )
    else:
        hist, bin_edges = detector.get_energy_hist(bins=None)
    
    # Calibrate bin edges
    calibrated_bin_edges = np.polyval(calibration_coeffs, bin_edges)
    xvals = np.diff(calibrated_bin_edges) / 2 + calibrated_bin_edges[:-1]
    
    # Plot histogram
    default_hist_kwargs = {'alpha': 0.7, 'label': 'Histogram'}
    if hist_kwargs:
        default_hist_kwargs.update(hist_kwargs)
    ax.stairs(hist, calibrated_bin_edges, **default_hist_kwargs)
    
    # Fit and plot peaks
    peak_energies = np.atleast_1d(peak_energies)
    all_fit_parameters = []
    
    # Group peaks by overlap threshold
    if len(peak_energies) > 1 and np.max(peak_energies) - np.min(peak_energies) > threshold_overlap:
        # Process peaks individually
        peak_groups = [[p] for p in peak_energies]
    else:
        # Process all peaks together
        peak_groups = [list(peak_energies)]
    
    default_fit_kwargs = {'linewidth': 2}
    if fit_kwargs:
        default_fit_kwargs.update(fit_kwargs)
    
    default_fill_kwargs = {'alpha': 0.3, 'label': 'Peak Area'}
    if fill_kwargs:
        default_fill_kwargs.update(fill_kwargs)
    
    fill_label_added = False
    
    for i, peak_group in enumerate(peak_groups):
        search_start = np.argmin(
            np.abs((peak_group[0] - search_width / len(peak_group)) - xvals)
        )
        search_end = np.argmin(
            np.abs((peak_group[-1] + search_width / len(peak_group)) - xvals)
        )
        
        # Build initial guess
        guess_parameters = [0, 0]
        for peak_erg in peak_group:
            peak_ind = np.argmin(np.abs(peak_erg - xvals))
            guess_parameters += [
                hist[peak_ind],
                peak_erg,
                search_width / (3 * len(peak_group)),
            ]
        
        if plot_initial_guess:
            ax.plot(
                xvals[search_start:search_end],
                gauss(xvals[search_start:search_end], *guess_parameters),
                '--',
                label=f'Initial Guess (Peak {i+1})' if len(peak_groups) > 1 else 'Initial Guess',
                alpha=0.5
            )
        
        # Fit the peak(s)
        try:
            parameters, covariance = curve_fit(
                gauss,
                xvals[search_start:search_end],
                hist[search_start:search_end],
                p0=guess_parameters,
            )
            all_fit_parameters.append(parameters)
            
            # Plot fitted curve
            ax.plot(
                xvals[search_start:search_end],
                gauss(xvals[search_start:search_end], *parameters),
                label=f'Fitted Curve (Peak {i+1})' if len(peak_groups) > 1 else 'Fitted Curve',
                **default_fit_kwargs
            )
            
            # Fill peak area for each peak in the group
            if plot_peak_area:
                for j in range(len(peak_group)):
                    # Extract individual peak parameters
                    mean = parameters[2 + 3 * j + 1]
                    sigma = np.abs(parameters[2 + 3 * j + 2])
                    
                    # Peak bounds at ±3 sigma
                    peak_start = np.argmin(np.abs((mean - 3 * sigma) - xvals))
                    peak_end = np.argmin(np.abs((mean + 3 * sigma) - xvals))
                    
                    peak_x = xvals[peak_start:peak_end]
                    
                    # Individual peak Gaussian curve
                    peak_params = [parameters[0], parameters[1], parameters[2 + 3 * j], mean, sigma]
                    peak_curve = gauss(peak_x, *peak_params)
                    
                    # Linear baseline (trap_cutoff)
                    baseline = parameters[0] + parameters[1] * peak_x
                    
                    # Fill between the Gaussian and the baseline
                    current_fill_kwargs = default_fill_kwargs.copy()
                    if fill_label_added:
                        current_fill_kwargs.pop('label', None)
                    else:
                        fill_label_added = True
                    
                    ax.fill_between(
                        peak_x,
                        baseline,
                        peak_curve,
                        **current_fill_kwargs
                    )
                    
                    # Plot the baseline
                    ax.plot(peak_x, baseline, 'k--', linewidth=1, alpha=0.5)
                    
        except RuntimeError as e:
            warnings.warn(f"Could not fit peak group {peak_group}: {e}")
            all_fit_parameters.append(None)
    
    ax.set_xlabel('Energy (keV)')
    ax.set_ylabel('Counts')
    ax.legend()
    if plot_title:
        ax.set_title(plot_title)
    
    return fig, ax, all_fit_parameters


def get_channel(filename, directory):
    """
    Extract the channel number from a given filename string.

    Parameters
    ----------
    filename : str
        The input filename string containing the channel information.
    directory : str
        The directory containing the filename.

    Returns
    -------
    int
        The extracted channel number.

    Example
    -------
        The input filename string containing the channel information.
        Should look something like : "Data_CH<channel_number>@V...CSV"

    Returns
    -------
    int
        The extracted channel number.

    Example
    -------
    >>> get_channel("Data_CH4@V1725_292_Background_250322.CSV", "UNFILTERED/")
    4
    >>> get_channel("DataR_CH5@V1725_292_Background_250322.CSV", "RAW/")
    5
    """
    # first determine if directory is for RAW or UNFILTERED data
    # as they will have different nameing conventions
    stem = Path(directory).stem
    if "RAW" in stem:
        return int(filename.split("@")[0][8:])
    else:
        return int(filename.split("@")[0][7:])


def sort_compass_files(directory: str, filetype=".csv") -> dict:
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
        if filename.lower().endswith(filetype):
            print(f"Found data file: {filename}")
            ch = get_channel(filename, directory)
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
    """Obtains count start and stop time from the run.info or *_info.txt file.
    Some versions of CoMPASS output a run.info file, while others output a *_info.txt file.
    This function checks for both and reads the time information from the file that exists."""

    run_info_file = Path(directory).parent / "run.info"
    info_txt_file = Path(directory).parent / f"{Path(directory).parent.stem}_info.txt"

    print("Hello world from get_start_stop_time!")
    print(f"Looking for run.info file at {run_info_file}")
    print(f"Looking for info.txt file at {info_txt_file}")

    if run_info_file.exists():
        start_time, stop_time = get_start_stop_time_from_run_info(run_info_file)
    elif info_txt_file.exists():
        print("Hello world from get_start_stop_time_from_info_txt!")
        start_time, stop_time = get_start_stop_time_from_info_txt(info_txt_file)
    else:
        raise FileNotFoundError(
            f"Could not find run.info file in parent directory {Path(directory).parent}"
        )
    return start_time, stop_time


def get_start_stop_time_from_run_info(info_file: Path) -> Tuple[datetime.datetime, datetime.datetime]:
    """Obtains count start and stop time from the run.info file."""
    time_format = "%Y/%m/%d %H:%M:%S.%f%z"
    with open(info_file, "r") as file:
        lines = file.readlines()

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


def get_start_stop_time_from_info_txt(info_txt_file: Path, tz=ZoneInfo("America/New_York")) -> Tuple[datetime.datetime, datetime.datetime]:
    """Obtains count start and stop time from the *_info.txt file.
    
    Args:
        info_txt_file: Path to the info.txt file
        tz: Timezone info for the datetime objects
        
    Returns:
        Tuple of (start_time, stop_time) as timezone-aware datetime objects (America/New_York)
    """
    
    time_format = "%a %b %d %H:%M:%S %Y"
    
    with open(info_txt_file, "r", encoding="latin-1") as file:
        lines = file.readlines()

    start_time, stop_time = None, None
    for line in lines:
        if line.startswith("Start time = "):
            time_string = line.split(" = ", 1)[1].strip()
            start_time = datetime.datetime.strptime(time_string, time_format)
            start_time = start_time.replace(tzinfo=tz)
        elif line.startswith("Stop time = "):
            time_string = line.split(" = ", 1)[1].strip()
            stop_time = datetime.datetime.strptime(time_string, time_format)
            stop_time = stop_time.replace(tzinfo=tz)

    if None in (start_time, stop_time):
        raise ValueError(f"Could not find 'Start time' or 'Stop time' in file {info_txt_file}.")
    
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


def get_live_time_from_info_txt(info_txt_file: Path, channel: int) -> Tuple[float, float]:
    """
    Gets live and real count time in seconds from Compass *_info.txt file.
    Live time is defined as the difference between the actual time that
    a count is occurring and the "dead time," in which the output of detector
    pulses is saturated such that additional signals cannot be processed.
    
    Args:
        info_txt_file: Path to the info.txt file
        channel: Channel number (e.g., 0 for CH0@, 1 for CH1@)
        
    Returns:
        Tuple of (live_count_time, real_count_time) in seconds
    """
    with open(info_txt_file, "r", encoding="latin-1") as file:
        lines = file.readlines()

    channel_header = f"CH{channel}@"
    in_channel_section = False
    live_count_time = None
    real_count_time = None

    for line in lines:
        if line.startswith(channel_header):
            in_channel_section = True
            continue
        elif in_channel_section and line.startswith("CH") and "@" in line:
            # We've reached the next channel section, stop searching
            break
        elif in_channel_section:
            if "Real time = " in line:
                # Extract time string like "0:03:14.665"
                match = re.search(r"Real time = (\d+:\d{2}:\d{2}\.\d+)", line)
                if match:
                    real_count_time = _parse_time_to_seconds(match.group(1))
            if "Live time = " in line:
                match = re.search(r"Live time = (\d+:\d{2}:\d{2}\.\d+)", line)
                if match:
                    live_count_time = _parse_time_to_seconds(match.group(1))
        
        if live_count_time is not None and real_count_time is not None:
            break

    if live_count_time is None or real_count_time is None:
        raise ValueError(
            f"Could not find Real time or Live time for channel {channel} in file {info_txt_file}."
        )

    return live_count_time, real_count_time


def _parse_time_to_seconds(time_str: str) -> float:
    """
    Parse a time string in format "H:MM:SS.mmm" to seconds.
    
    Args:
        time_str: Time string like "0:03:14.665"
        
    Returns:
        Time in seconds as a float
    """
    parts = time_str.split(":")
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def get_spectrum_nbins(settings_file: str) -> int:
    """
    Read a settings.xml file and extract the number of spectrum bins
    from the SRV_PARAM_CH_SPECTRUM_NBINS parameter.
    
    Args:
        settings_file: Path to the settings.xml file
        
    Returns:
        The number of bins as an integer (e.g., 16384 from "BINS_16384")
    """
    tree = ET.parse(settings_file)
    root = tree.getroot()
    
    # Find entry with the matching key
    for entry in root.iter('entry'):
        key_elem = entry.find('key')
        if key_elem is not None and key_elem.text == 'SRV_PARAM_CH_SPECTRUM_NBINS':
            # Get the first value element (contains the actual value)
            value_container = entry.find('value')
            if value_container is not None:
                value_elem = value_container.find('value')
                if value_elem is not None and value_elem.text:
                    # Extract number from "BINS_16384" format
                    match = re.search(r'BINS_(\d+)', value_elem.text)
                    if match:
                        return int(match.group(1))
    print(f"SRV_PARAM_CH_SPECTRUM_NBINS not found in settings file {settings_file}. Defaulting to None bins.")
    return None
