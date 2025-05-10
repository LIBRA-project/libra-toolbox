import numpy as np
from numpy.typing import NDArray
import os
from pathlib import Path
import pandas as pd
from typing import Tuple, Dict, List, Union
import datetime
import uproot
import glob

import warnings
from libra_toolbox.neutron_detection.activation_foils.calibration import (
    CheckSource,
    na22,
    co60,
    ba133,
    mn54,
)

from scipy.signal import find_peaks
from scipy.optimize import curve_fit


class Detector:
    events: NDArray[Tuple[float, float]]  # type: ignore # Array of (time in ps, energy) pairs
    channel_nb: int
    live_count_time: float
    real_count_time: float

    def __init__(self, channel_nb) -> None:
        """
        Initialize a Detector object.
        Args:
            channel_nb: channel number of the detector
        """
        self.channel_nb = channel_nb
        self.events = np.empty((0, 2))  # Initialize as empty 2D array with 2 columns
        self.live_count_time = None
        self.real_count_time = None

    def get_energy_hist(
        self, bins: Union[None, NDArray[np.float64]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the energy histogram of the detector events.
        Args:
            bins: number of bins or "double" to use half the max energy as bin size
        Returns:
            Tuple of histogram values and bin edges
        """

        energy_values = self.events[:, 1].copy()
        time_values = self.events[:, 0].copy()

        # sort data based on timestamp
        inds = np.argsort(time_values)
        time_values = time_values[inds]
        energy_values = energy_values[inds]

        energy_values = np.nan_to_num(energy_values, nan=0)

        if bins is None:
            bins = int(np.nanmax(energy_values)) + 1

        return np.histogram(energy_values, bins=bins)

    def get_energy_hist_background_substract(
        self,
        background_detector: "Detector",
        bins: Union[NDArray[np.float64], None] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ps_to_seconds = 1e-12
        raw_hist, raw_bin_edges = self.get_energy_hist(bins=bins)
        background_times = background_detector.events[:, 0].copy()
        background_energies = background_detector.events[:, 1].copy()

        if self.real_count_time < background_detector.real_count_time:
            # get background counts for the duration of the sample count
            end_ind = np.nanargmin(
                np.abs(
                    self.real_count_time / ps_to_seconds
                    - (background_times - background_times[0])
                )
            )
            b_hist, _ = np.histogram(
                background_energies[: end_ind + 1],
                bins=raw_bin_edges,
            )
        else:
            b_hist, _ = np.histogram(background_energies, bins=raw_bin_edges)
            b_hist = b_hist * (
                self.real_count_time / background_detector.real_count_time
            )

        hist_background_substracted = raw_hist - b_hist

        return hist_background_substracted, raw_bin_edges


class Measurement:
    start_time: datetime.datetime
    stop_time: datetime.datetime
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


class CheckSourceMeasurement(Measurement):
    check_source: CheckSource

    def get_expected_activity(self) -> float:
        """
        Calculates the expected activity of a check source given the
        half-life and the date of the measurement.
        The expected activity is calculated using the formula:
        .. math:: A(t) = A_0 e^{-\\lambda t}

        where :math:`A_0` is the initial activity, :math:`\\lambda` is the decay constant
        and :math:`t` is the time since the measurement date.

        Returns:
            the expected activity of the check source in Bq
        """
        decay_constant = np.log(2) / self.check_source.nuclide.half_life

        # Convert date to datetime if needed
        if isinstance(
            self.check_source.activity_date, datetime.date
        ) and not isinstance(self.check_source.activity_date, datetime.datetime):
            activity_datetime = datetime.datetime.combine(
                self.check_source.activity_date, datetime.time.min
            )
            # add a timezone
            activity_datetime = activity_datetime.replace(tzinfo=self.start_time.tzinfo)
        else:
            activity_datetime = self.check_source.activity_date

        time = (self.start_time - activity_datetime).total_seconds()
        act_expec = self.check_source.activity * np.exp(-decay_constant * time)
        return act_expec

    # should be a method of a class called CheckSourceMeasurement
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
        .. math:: \\eta = \\frac{A_{meas}}{A_{expec}}

        where :math:`A_{meas}` is the measured activity and :math:`A_{expec}` is the expected activity.
        The measured activity is calculated using the formula:
        .. math:: A_{meas} = \\frac{A_{peak}}{I \\cdot t_{live}}

        where :math:`A_{peak}` is the area of the peak, :math:`I` is the intensity of the check source
        and :math:`t_{live}` is the live count time of the detector.

        Args:
            background_measurement: _description_
            calibration_coeffs: _description_

        Returns:
            the detection efficiency
        """
        # find right background detector

        background_detector = [
            d for d in background_measurement.detectors if d.channel_nb == channel_nb
        ][0]
        check_source_detector = [
            d for d in self.detectors if d.channel_nb == channel_nb
        ][0]

        hist, bin_edges = check_source_detector.get_energy_hist_background_substract(
            background_detector, bins=None
        )

        calibrated_bin_bedges = np.polyval(calibration_coeffs, bin_edges)

        areas = get_multipeak_area(
            hist,
            calibrated_bin_bedges,
            self.check_source.nuclide.energy,
            search_width=search_width,
        )

        act_meas = np.array(areas) / (
            np.array(self.check_source.nuclide.intensity)
            * check_source_detector.live_count_time
        )
        act_meas_err = np.sqrt(np.array(areas)) / (
            np.array(self.check_source.nuclide.intensity)
            * check_source_detector.live_count_time
        )

        act_expec = self.get_expected_activity()

        detection_efficiency = act_meas / act_expec

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


def get_calibration_data(
    check_source_measurements: List[CheckSourceMeasurement],
    background_measurement: Measurement,
    channel_nb: int,
):
    background_detector = [
        detector
        for detector in background_measurement.detectors
        if detector.channel_nb == detector.channel_nb
    ][0]

    calibration_energies = []
    calibration_channels = []

    for measurement in check_source_measurements:
        for detector in measurement.detectors:
            if detector.channel_nb != channel_nb:
                continue

            sample = measurement.name[:-2]

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


# NOTE is this function really needed?
def get_calibration_curve(
    check_source_measurements: List[CheckSourceMeasurement],
    background_measurement: Measurement,
    channel_nb: int,
):

    calibration_channels, calibration_energies = get_calibration_data(
        check_source_measurements,
        background_measurement,
        channel_nb,
    )

    # linear fit for calibration curve
    coeff = np.polyfit(
        calibration_channels,
        calibration_energies,
        1,
    )

    return coeff


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
        gross_area = np.trapezoid(
            gauss(xvals[peak_start:peak_end], *peak_params),
            x=xvals[peak_start:peak_end],
        )

        # Cut off trapezoidal area due to compton scattering and noise
        trap_cutoff_area = np.trapezoid(
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

            # only the first file has a header
            if i == 0:
                header = 0
            else:
                header = None

            csv_file_path = os.path.join(directory, filename)

            df = pd.read_csv(csv_file_path, delimiter=";", header=header)

            # read the header and store in names
            if i == 0:
                names = df.columns.values
            else:
                # apply the column names if not the first file
                df.columns = names

            time_data = df["TIMETAG"].to_numpy()
            energy_data = df["ENERGY"].to_numpy()

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
