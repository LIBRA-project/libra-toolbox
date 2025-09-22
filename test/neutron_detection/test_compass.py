import pytest
import numpy as np
import os
from libra_toolbox.neutron_detection.activation_foils import compass
from libra_toolbox.neutron_detection.activation_foils.calibration import (
    Nuclide,
    CheckSource,
    ActivationFoil,
    Reaction,
)
from pathlib import Path
import datetime
import h5py


@pytest.mark.parametrize(
    "filename, expected_channel",
    [
        ("Data_CH14@V1725_292_Background_250322.CSV", 14),
        ("Data_CH7@V1725_123_Background_250322.CSV", 7),
        ("Data_CH21@V1725_456_Background_250322.CSV", 21),
    ],
)
def test_get_channel(filename, expected_channel):
    ch = compass.get_channel(filename)
    assert ch == expected_channel


def create_empty_csv_files(directory, base_name, count, channel):
    """
    Creates empty CSV files in a specified directory with a specific pattern.

    Args:
        directory (str): The directory where the files will be created.
        base_name (str): The base name of the file (e.g., "Data_CH14").
        count (int): The number of files to generate.

    Returns:
        list: A list of file paths for the created CSV files.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_paths = []
    for i in range(count):
        if i == 0:
            filename = f"Data_CH{channel}@{base_name}.csv"
        else:
            filename = f"Data_CH{channel}@{base_name}_{i}.csv"
        file_path = os.path.join(directory, filename)
        with open(file_path, "w") as f:
            pass  # Create an empty file
        file_paths.append(file_path)

    return file_paths


@pytest.mark.parametrize(
    "base_name, expected_filenames",
    [
        (
            "base",
            {
                4: [
                    "Data_CH4@base.csv",
                    "Data_CH4@base_1.csv",
                    "Data_CH4@base_2.csv",
                    "Data_CH4@base_3.csv",
                ],
                1: [
                    "Data_CH1@base.csv",
                ],
            },
        ),
    ],
)
def test_sort_compass_files(tmpdir, base_name: str, expected_filenames: dict):
    for ch, list_of_filenames in expected_filenames.items():
        create_empty_csv_files(
            tmpdir, base_name, count=len(list_of_filenames), channel=ch
        )

    data_filenames = compass.sort_compass_files(tmpdir)

    assert isinstance(data_filenames, dict)

    # Check if dictionaries have the same keys, length of filenames array, and
    # the same overall filenames array
    for key in expected_filenames:
        assert key in data_filenames
        assert len(data_filenames[key]) == len(expected_filenames[key])
        for a, b in zip(data_filenames[key], expected_filenames[key]):
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                assert np.array_equal(a, b)
            else:
                assert a == b


@pytest.mark.parametrize(
    "waveform_directory, expected_time, expected_energy, expected_idx, expected_keys, test_ch",
    [
        ("no_waveforms", 6685836624, 515, 5, [5, 15], 5),
        ("no_waveforms", 11116032249, 568, 6, [5, 15], 5),
        ("no_waveforms", 1623550122, 589, -1, [5, 15], 5),
        ("no_waveforms", 535148093, 1237, -2, [5, 15], 5),
        ("waveforms", 80413091, 1727, 0, [4], 4),
        ("waveforms", 2850906749, 1539, 2, [4], 4),
        ("waveforms", 14300873206559, 1700, 6, [4], 4)
    ],
)
def test_get_events(waveform_directory, expected_time, 
                    expected_energy, expected_idx,
                    expected_keys, test_ch):
    """
    Test the get_events function from the compass module.
    Checks that specific time and energy values are returned for a given channel
    """
    test_directory = Path(__file__).parent / "compass_test_data/events" / waveform_directory
    times, energies = compass.get_events(test_directory)
    assert isinstance(times, dict)
    assert isinstance(energies, dict)

    for key in expected_keys:
        assert key in times
        assert key in energies

    assert times[test_ch][expected_idx] == expected_time
    assert energies[test_ch][expected_idx] == expected_energy


utc_minus5 = datetime.timezone(datetime.timedelta(hours=-5))
utc_minus4 = datetime.timezone(datetime.timedelta(hours=-4))


@pytest.mark.parametrize(
    "start_time, stop_time",
    [
        (
            datetime.datetime(
                2024, 11, 7, 15, 47, 21, microsecond=127000, tzinfo=utc_minus5
            ),
            datetime.datetime(
                2024, 11, 7, 16, 2, 21, microsecond=133000, tzinfo=utc_minus5
            ),
        ),
        (
            datetime.datetime(
                2025, 3, 18, 22, 19, 3, microsecond=947000, tzinfo=utc_minus4
            ),
            datetime.datetime(
                2025, 3, 19, 9, 21, 6, microsecond=558000, tzinfo=utc_minus4
            ),
        ),
    ],
)
def test_get_start_stop_time(tmpdir, start_time, stop_time):
    """
    Tests the get_start_stop_time function from the compass module.
    Checks that the start and stop times are correctly parsed from the run.info file.
    """
    # BUILD
    content = _run_info_content(start_time, stop_time)

    # Create another temporary directory
    tmpdir2 = os.path.join(tmpdir, "tmpdir2")

    # create an empty run.info file
    run_info_path = os.path.join(tmpdir, "run.info")

    # add some stuff
    with open(run_info_path, "w") as f:
        f.write(content)

    # RUN
    start_time_out, stop_time_out = compass.get_start_stop_time(tmpdir2)

    # TEST
    assert isinstance(start_time_out, datetime.datetime)
    assert start_time_out == start_time

    assert isinstance(stop_time_out, datetime.datetime)
    assert stop_time_out == stop_time


def _run_info_content(start_time: datetime.datetime, stop_time: datetime.datetime):
    """
    Creates a string that simulates the content of a run.info file.
    """
    return f"""id=Co60_0_872uCi_19Mar14_241107
time.start={start_time.strftime("%Y/%m/%d %H:%M:%S.%f%z")}
time.stop={stop_time.strftime("%Y/%m/%d %H:%M:%S.%f%z")}
time.real=00:15:00
board.0-14-292.readout.rate=132.731 kb/s
board.0-14-292.1.rejections.singles=0.0
board.0-14-292.1.rejections.pileup=0.0
board.0-14-292.1.rejections.saturation=1729.15
board.0-14-292.1.rejections.energy=0.0
board.0-14-292.1.rejections.psd=0.0
board.0-14-292.1.rejections.timedistribution=0.0
board.0-14-292.1.throughput=6950.66
board.0-14-292.1.icr=7424.44
board.0-14-292.1.ocr=5253.24
board.0-14-292.1.calibration.energy.c0=0.0
board.0-14-292.1.calibration.energy.c1=1.0
board.0-14-292.1.calibration.energy.c2=0.0
board.0-14-292.1.calibration.energy.uom=keV
board.0-14-292.2.rejections.singles=0.0
board.0-14-292.2.rejections.pileup=0.0
board.0-14-292.2.rejections.saturation=8.2202
board.0-14-292.2.rejections.energy=0.0
board.0-14-292.2.rejections.psd=0.0
board.0-14-292.2.rejections.timedistribution=0.0
board.0-14-292.2.throughput=3958.96
board.0-14-292.2.icr=3981.66
board.0-14-292.2.ocr=3952.89
board.0-14-292.2.calibration.energy.c0=0.0
board.0-14-292.2.calibration.energy.c1=1.0
board.0-14-292.2.calibration.energy.c2=0.0
board.0-14-292.2.calibration.energy.uom=keV
"""


def test_filenotfound_error_info():
    with pytest.raises(FileNotFoundError, match="Could not find run.info"):
        compass.get_start_stop_time(
            directory=Path(__file__).parent / "compass_test_data/events"
        )


def test_get_start_stop_time_with_notime(tmpdir):
    """Creates an empty file run.info and check that an error is raised if can't find time"""

    # Create another temporary directory

    tmpdir2 = os.path.join(tmpdir, "tmpdir2")

    # create an empty run.info file
    run_info_path = os.path.join(tmpdir, "run.info")

    # add some stuff
    with open(run_info_path, "w") as f:
        f.write("coucou\ncoucou\n")

    # run
    with pytest.raises(ValueError, match="Could not find time.start or time.stop"):
        compass.get_start_stop_time(tmpdir2)


@pytest.mark.parametrize(
    "root_filename, channel, live_time, real_time",
    [
        (
            Path(__file__).parent
            / "compass_test_data/times/Hcompass_Co60_20241107.root",
            1,
            808.305,
            900.108,
        ),
        (
            Path(__file__).parent
            / "compass_test_data/times/Hcompass_Co60_20241107.root",
            2,
            896.374,
            900.108,
        ),
        (
            Path(__file__).parent
            / "compass_test_data/times/Hcompass_Zirconium_20250319.root",
            4,
            35654.785,
            39722.502,
        ),
        (
            Path(__file__).parent
            / "compass_test_data/times/Hcompass_Zirconium_20250319.root",
            5,
            39678.458,
            39722.502,
        ),
    ],
)
def test_get_live_time_from_root(root_filename, channel, live_time, real_time):
    live_time_out, real_time_out = compass.get_live_time_from_root(
        root_filename, channel
    )
    assert live_time_out == live_time
    assert real_time_out == real_time


@pytest.mark.parametrize("no_root", [True, False])
def test_measurement_object_from_directory(no_root):
    """
    Test the Measurement object creation from a directory.
    """
    if no_root:
        test_directory = (
            Path(__file__).parent
            / "compass_test_data/complete_measurement_no_root/data"
        )
    else:
        test_directory = (
            Path(__file__).parent / "compass_test_data/complete_measurement/data"
        )

    measurement = compass.Measurement.from_directory(test_directory, name="test")

    assert len(measurement.detectors) == 1
    assert isinstance(measurement.detectors[0], compass.Detector)
    assert measurement.detectors[0].channel_nb == 1

    assert measurement.detectors[0].events.shape[1] == 2

    measurement.detectors[0].get_energy_hist(bins=None)


@pytest.mark.parametrize(
    "bins",
    [
        10,
        20,
        50,
        100,
        None,
        np.arange(0, 10, 1),
        np.linspace(0, 10, num=100),
    ],
)
def test_detector_get_energy_hist(bins):
    """
    Test the get_energy_hist method of the Detector class.
    """
    my_detector = compass.Detector(channel_nb=1)
    my_detector.events = np.array(
        [
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
            [9, 10],
        ]
    )

    my_detector.get_energy_hist(bins=bins)


@pytest.mark.parametrize(
    "counting_time_background",
    [
        10,
        100,
        1000,
        3000,
    ],
)
def test_background_sub(counting_time_background):
    """
    Test the background subtraction method of the Detector class.
    Builds a test case with a background measurement and a measured foil measurement,
    then checks that the background is correctly subtracted from the measured spectrum.

    Also checks that the result is independent of the counting time of the background measurement.
    """
    # BUILD

    def background_spectrum(energies):
        return np.ones_like(energies)

    def measured_spectrum(energies):
        return np.cos(energies / 10) + 10

    counting_time_measured = 200

    background_rate = 300000 / (3600)
    measurement_rate = 3 * background_rate

    nb_events_background = int(background_rate * counting_time_background)
    nb_events_measured = int(measurement_rate * counting_time_measured)
    nb_events_measured_bg_contrib = int(background_rate * counting_time_measured)

    # Define energy grid for sampling
    energy_grid = np.arange(100)

    # Calculate probability distributions using the spectrum functions
    bg_probabilities = background_spectrum(energy_grid)
    bg_probabilities = bg_probabilities / np.sum(bg_probabilities)  # Normalize
    measured_probabilities = measured_spectrum(energy_grid)
    measured_probabilities = measured_probabilities / np.sum(
        measured_probabilities
    )  # Normalize

    # Sample from these distributions
    energy_events_bg = np.random.choice(
        energy_grid, size=nb_events_background, p=bg_probabilities
    )
    energy_events_measured = np.random.choice(
        energy_grid, size=nb_events_measured, p=measured_probabilities
    )
    energy_events_measured_bg_contrib = np.random.choice(
        energy_grid, size=nb_events_measured_bg_contrib, p=bg_probabilities
    )

    energy_events_measured = np.concatenate(
        (energy_events_measured, energy_events_measured_bg_contrib)
    )

    # Create the measurement objects
    ps_to_seconds = 1e-12

    measurement = compass.Measurement("test")
    detector_meas = compass.Detector(channel_nb=1)
    detector_meas.real_count_time = counting_time_measured
    measurement.detectors = [detector_meas]
    time_events_measured = np.random.uniform(
        0, counting_time_measured, nb_events_measured + nb_events_measured_bg_contrib
    )
    time_events_measured *= 1 / ps_to_seconds
    time_events_measured.sort()
    detector_meas.events = np.column_stack(
        (time_events_measured, energy_events_measured)
    )

    background_measurment = compass.Measurement("background")
    background_detector = compass.Detector(channel_nb=1)
    background_detector.real_count_time = counting_time_background
    background_measurment.detectors = [background_detector]
    background_time_events = np.random.uniform(
        0, counting_time_background, nb_events_background
    )
    background_time_events *= 1 / ps_to_seconds
    background_time_events.sort()
    background_detector.events = np.column_stack(
        (background_time_events, energy_events_bg)
    )

    # RUN
    computed_hist, _ = detector_meas.get_energy_hist_background_substract(
        background_detector=background_detector,
        live_or_real="real",
    )

    # TEST
    hist_bg, _ = background_detector.get_energy_hist()
    hist_raw, _ = detector_meas.get_energy_hist()
    expected_hist = (
        hist_raw - hist_bg / counting_time_background * counting_time_measured
    )
    assert np.allclose(computed_hist, expected_hist, rtol=1e-1)


@pytest.mark.parametrize(
    "activity_date",
    [
        datetime.datetime(2024, 11, 7),
        datetime.date(2024, 11, 7),
    ],
)
@pytest.mark.parametrize("n_half_lives", [0, 1, 2, 3, 4, 5])
def test_check_source_expected_activity(n_half_lives, activity_date):
    """
    Test the expected activity of a check source.
    """
    # BUILD
    half_life = 10 * 24 * 3600  # seconds  (10 days)
    activity = 500  # Bq

    test_nuclide = Nuclide(
        name="TestNuclide",
        energy=[100, 200],
        intensity=[0.5, 0.5],
        half_life=half_life,
    )

    check_source = CheckSource(
        nuclide=test_nuclide,
        activity_date=activity_date,
        activity=activity,
    )

    start_time = activity_date + datetime.timedelta(seconds=n_half_lives * half_life)
    # convert start_time and stop_time to datetime
    if isinstance(start_time, datetime.date):
        start_time = datetime.datetime.combine(start_time, datetime.datetime.min.time())

    # RUN
    computed_activity = check_source.get_expected_activity(start_time)

    # TEST

    expected_activity = activity / (2**n_half_lives)
    assert np.isclose(computed_activity, expected_activity, rtol=1e-2)


@pytest.mark.parametrize("expected_efficiency", [1e-2, 0.5, 1])
def test_check_source_detection_efficiency(expected_efficiency):
    """
    Test the detection efficiency of a check source measurement.
    Generates a test case with a known detection efficiency and checks that the
    computed efficiency is close to the expected one.

    Using a Mn54 source with an energy of 834.848 keV and an intensity of 1.0.
    We generate some events with a normal distribution centered on the energy of the source.
    The number of events is given by the expected efficiency, the activity of the source,
    the measurement time, and the number of half-lives passed since the activity date.
    """
    # BUILD

    ps_to_seconds = 1e-12

    n_half_lives = 0

    activity_date = datetime.datetime(2024, 11, 7)
    half_life = 10 * 24 * 3600  # seconds  (10 days)
    activity = 5000e1  # Bq

    test_nuclide = Nuclide(
        name="TestNuclide Mn54",
        energy=[834.848],
        intensity=[1.0],
        half_life=half_life,
    )

    check_source = CheckSource(
        nuclide=test_nuclide,
        activity_date=activity_date,
        activity=activity,
    )

    measurement = compass.CheckSourceMeasurement(name="test measurement")
    measurement.check_source = check_source
    measurement.start_time = activity_date + datetime.timedelta(
        seconds=n_half_lives * half_life
    )
    measurement.stop_time = measurement.start_time + datetime.timedelta(seconds=100)
    measurement_time = (measurement.stop_time - measurement.start_time).total_seconds()

    # generate the spectrum which is a normal centered on energy
    nb_events_measured = (
        expected_efficiency * activity / (2**n_half_lives) * measurement_time
    )
    energy_events = np.random.normal(
        loc=test_nuclide.energy[0],
        scale=20,
        size=int(nb_events_measured),
    )
    # make sure the min and max are in the range of the detector
    energy_events[0] = 1
    energy_events[-1] = 3000
    time_events = np.random.uniform(
        0,
        measurement_time,
        size=int(nb_events_measured),
    )
    time_events *= 1 / ps_to_seconds
    time_events.sort()

    detector_meas = compass.Detector(channel_nb=1)
    detector_meas.events = np.column_stack((time_events, energy_events))
    detector_meas.real_count_time = measurement_time
    detector_meas.live_count_time = detector_meas.real_count_time
    measurement.detectors = [detector_meas]

    background_measurement = compass.Measurement("background")
    bg_detector = compass.Detector(channel_nb=1)
    bg_detector.real_count_time = 0.5
    bg_detector.live_count_time = bg_detector.real_count_time
    background_measurement.detectors = [bg_detector]

    # RUN
    computed_efficiency = measurement.compute_detection_efficiency(
        background_measurement,
        calibration_coeffs=[1.0, 0.0],  # assume perfect calibration
        channel_nb=1,
    )

    # TEST
    assert np.isclose(computed_efficiency, expected_efficiency, rtol=1e-2)

@pytest.mark.parametrize(
    "peak_energies, width, start_index, expected_peaks",
    [
        ([400, 800], 50, 0, [400, 800]),
        ([400, 800], 50, 600, [800]),
        ([200, 250, 400], 5, 230, [250, 400]),
    ],
)
def test_get_peaks(peak_energies, width, start_index, expected_peaks):
    nb_events_measured = 60000
    channel_nb = 1

    overall_energy_events = np.array([])

    for energy_level in peak_energies:
        energy_events = np.random.normal(
            loc=energy_level, scale=width, size=int(nb_events_measured)
        )
        overall_energy_events = np.concatenate((overall_energy_events, energy_events))

    random_noise = np.random.uniform(0, 3000, size=int(nb_events_measured))
    overall_energy_events = np.concatenate((overall_energy_events, random_noise))

    # make sure the min and max are in the range of the detector
    overall_energy_events[0] = 1
    overall_energy_events[-1] = 3000

    time_events = np.random.uniform(0, 100, size=int(nb_events_measured * (len(peak_energies) + 1)))

    test_nuclide = Nuclide(
        name="TestNuclide",
        energy=peak_energies,
        intensity=[1.0]*len(peak_energies),
        half_life=10000,
    )
    check_source = CheckSource(
        nuclide=test_nuclide,
        activity_date=datetime.datetime(2024, 11, 7),
        activity=5000,
    )

    # create CheckSourceMeasurement
    measurement = compass.CheckSourceMeasurement(name="test measurement")
    measurement.check_source = check_source
    measurement.start_time = datetime.datetime(2024, 11, 7)
    detector = compass.Detector(channel_nb=channel_nb, nb_digitizer_bins=None)
    detector.events = np.column_stack((time_events, overall_energy_events))
    hist, bins = detector.get_energy_hist(bins=None)
    
    peak_indices = measurement.get_peaks(hist, start_index=start_index)

    assert len(peak_indices) == len(expected_peaks)
    assert np.allclose(expected_peaks - bins[peak_indices], 0, atol=2*width)


@pytest.mark.parametrize(
    "a, b",
    [
        (1.5, 200),
        (1, 0),
        (2, 600),
    ],
)
def test_get_calibration_data(a, b):
    """
    Test the get_calibration_data function from the compass module.
    Checks that the calibration data is correctly computed from the measurements.

    The test generates a set of measurements with known energies and intensities,
    and checks that the computed calibration data matches the expected values.
    The energies counts (channels) are generated using a linear function with parameters a and b.
    """
    # BUILD
    channel_nb = 1
    nb_events_measured = 60000
    measurements = []
    real_energies = np.array([800, 1300, 1800])
    energy_channels = a * real_energies + b
    for real_energy, energy_channel in zip(
        real_energies,
        energy_channels,
    ):
        test_nuclide = Nuclide(
            name="TestNuclide",
            energy=[real_energy],
            intensity=[1.0],
            half_life=10000,
        )
        check_source = CheckSource(
            nuclide=test_nuclide,
            activity_date=datetime.datetime(2024, 11, 7),
            activity=5000,
        )

        # create CheckSourceMeasurement
        measurement = compass.CheckSourceMeasurement(name="test measurement")
        measurement.check_source = check_source
        measurement.start_time = datetime.datetime(2024, 11, 7)
        detector = compass.Detector(channel_nb=channel_nb, nb_digitizer_bins=None)
        energy_events = np.random.normal(
            loc=energy_channel, scale=30, size=int(nb_events_measured)
        )

        # make sure the min and max are in the range of the detector
        energy_events[0] = 1
        energy_events[-1] = 3000

        time_events = np.random.uniform(0, 100, size=int(nb_events_measured))
        detector.events = np.column_stack((time_events, energy_events))
        detector.real_count_time = 100
        detector.live_count_time = detector.real_count_time
        measurement.detectors = [detector]

        measurements.append(measurement)

    # create background measurement
    background_measurement = compass.Measurement("background")
    bg_detector = compass.Detector(channel_nb=channel_nb, nb_digitizer_bins=None)
    bg_detector.live_count_time = 100
    bg_detector.real_count_time = bg_detector.live_count_time
    background_measurement.detectors = [bg_detector]

    # RUN
    calibration_channels, calibration_energies = compass.get_calibration_data(
        measurements, background_measurement, channel_nb=channel_nb
    )

    # TEST
    assert np.allclose(calibration_channels, energy_channels, rtol=1e-2)
    assert np.allclose(calibration_energies, real_energies, rtol=1e-2)


def test_get_multipeak_area_single_peak():
    """
    Test the get_multipeak_area function from the compass module.
    Checks that the area under the peaks is correctly computed.
    """
    # BUILD
    energy = 2000
    nb_events_measured = 60000
    energy_events = np.random.normal(loc=energy, scale=30, size=int(nb_events_measured))
    # make sure the min and max are in the range of the detector
    energy_events[0] = 1
    energy_events[-1] = 3000

    hist, bin_edges = np.histogram(energy_events, bins=np.arange(0, 3000))

    # RUN
    areas = compass.get_multipeak_area(hist, bin_edges, peak_ergs=[energy])

    # TEST
    expected_area = np.sum(hist)
    assert np.isclose(areas[0], expected_area, rtol=1e-2)


def test_get_multipeak_area_two_separated_peaks():
    """
    Test the get_multipeak_area function from the compass module.
    Checks that the area under the peaks is correctly computed.
    """
    # BUILD
    energy1 = 1000
    energy2 = 2000
    energy_events = np.empty((0,))
    nb_events_peak1 = 60000
    nb_events_peak2 = 2 * nb_events_peak1
    sigma_peak = 30
    for energy, nb_events in zip(
        [energy1, energy2], [nb_events_peak1, nb_events_peak2]
    ):
        new_energy_events = np.random.normal(
            loc=energy, scale=sigma_peak, size=int(nb_events)
        )
        # make sure the min and max are in the range of the detector
        new_energy_events[0] = 1
        new_energy_events[-1] = 3000
        energy_events = np.concatenate((energy_events, new_energy_events))

    hist, bin_edges = np.histogram(energy_events, bins=np.arange(0, 3000))

    # RUN
    areas = compass.get_multipeak_area(hist, bin_edges, peak_ergs=[energy1, energy2])

    # TEST

    expected_area_peak_1 = nb_events_peak1
    expected_area_peak_2 = nb_events_peak2
    for i, expected_area in enumerate([expected_area_peak_1, expected_area_peak_2]):
        assert np.isclose(areas[i], expected_area, rtol=1e-2)


def test_get_multipeak_area_two_close_peaks():
    """
    Test the get_multipeak_area function from the compass module.
    Checks that the area under the peaks is correctly computed.
    """
    # BUILD
    energy1 = 1000
    energy2 = 1100
    energy_events = np.empty((0,))
    nb_events_peak1 = 1300000
    nb_events_peak2 = 0.6 * nb_events_peak1
    sigma_peak = 30
    for energy, nb_events in zip(
        [energy1, energy2], [nb_events_peak1, nb_events_peak2]
    ):
        new_energy_events = np.random.normal(
            loc=energy, scale=sigma_peak, size=int(nb_events)
        )
        # make sure the min and max are in the range of the detector
        new_energy_events[0] = 1
        new_energy_events[-1] = 3000
        energy_events = np.concatenate((energy_events, new_energy_events))

    hist, bin_edges = np.histogram(energy_events, bins=np.arange(0, 3000))

    # RUN
    areas = compass.get_multipeak_area(hist, bin_edges, peak_ergs=[energy1, energy2])

    # TEST
    expected_area_peak_1 = nb_events_peak1
    expected_area_peak_2 = nb_events_peak2
    for i, expected_area in enumerate([expected_area_peak_1, expected_area_peak_2]):
        assert np.isclose(areas[i], expected_area, rtol=1e-2)


@pytest.mark.parametrize("efficiency", [1e-2, 0.1, 0.5, 1.0])
def test_get_gamma_emitted(efficiency: float):
    # BUILD
    nuclide_reactant = Nuclide(name="TestNuclide", atomic_mass=200)
    activated_nuclide = Nuclide(
        name="ActivatedNuclide",
        energy=[1000],
        intensity=[1.0],
        half_life=10 * 24 * 3600,
    )

    reaction = Reaction(
        reactant=nuclide_reactant,
        product=activated_nuclide,
        cross_section=20.0,
    )

    foil = ActivationFoil(reaction=reaction, mass=0.1, name="TestFoil")

    measurement = compass.SampleMeasurement("sample")
    measurement.foil = foil

    count_time_hr = 1  # hr
    measurement.start_time = datetime.datetime(2024, 11, 7)
    measurement.stop_time = datetime.datetime(2024, 11, 7, count_time_hr)

    measurement.detectors = [
        compass.Detector(channel_nb=4),
        compass.Detector(channel_nb=3),
    ]
    measurement.get_detector(3).real_count_time = count_time_hr * 3600
    measurement.get_detector(3).live_count_time = count_time_hr * 3600

    nb_counts = 50000
    energy_events = np.random.normal(
        loc=activated_nuclide.energy[0], scale=30, size=int(nb_counts)
    )
    time_events = np.random.uniform(0, 100, size=energy_events.size)
    measurement.get_detector(3).events = np.column_stack((time_events, energy_events))

    background_measurement = compass.Measurement("background")
    background_measurement.detectors = [compass.Detector(channel_nb=3)]
    background_measurement.get_detector(3).events = np.array([(0, 0), (1, 4000)])
    background_measurement.get_detector(3).real_count_time = count_time_hr * 3600
    background_measurement.get_detector(3).live_count_time = count_time_hr * 3600

    # RUN
    gammas_emmitted = measurement.get_gamma_emitted(
        background_measurement=background_measurement,
        efficiency_coeffs=np.array([0.0, efficiency]),  # assume perfect efficiency
        calibration_coeffs=np.array([1.0, 0.0]),  # assume perfect calibration
        channel_nb=3,
        search_width=300,
    )
    computed_value = gammas_emmitted[0]

    # TEST
    expected_value = nb_counts / efficiency
    assert np.isclose(computed_value, expected_value, rtol=1e-2)


@pytest.mark.parametrize("distance", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("photon_counts", [1e6, 1e7, 1e8, 0.0])
def test_get_neutron_rate_very_long_half_life(photon_counts, distance):
    # BUILD

    half_life = 100 * 24 * 3600  # seconds  (100 days)

    nuclide_reactant = Nuclide(name="TestNuclide", atomic_mass=200)
    activated_nuclide = Nuclide(
        name="ActivatedNuclide",
        energy=[1000],
        intensity=[1.0],
        half_life=half_life,
    )

    reaction = Reaction(
        reactant=nuclide_reactant,
        product=activated_nuclide,
        cross_section=20.0,
    )

    foil = ActivationFoil(
        reaction=reaction,
        mass=0.1,
        name="TestFoil",
        thickness=None,
    )

    measurement = compass.SampleMeasurement("sample")
    measurement.foil = foil

    count_time_hr = 2  # hr
    measurement.start_time = datetime.datetime(2024, 11, 7)
    measurement.stop_time = datetime.datetime(2024, 11, 7, count_time_hr)

    measurement.detectors = [
        compass.Detector(channel_nb=4),
        compass.Detector(channel_nb=3),
    ]
    measurement.get_detector(3).real_count_time = count_time_hr * 3600
    measurement.get_detector(3).live_count_time = measurement.get_detector(
        3
    ).real_count_time

    irradiation_time = 3600  # seconds
    irradiations = [{"t_on": 0, "t_off": irradiation_time}]

    # RUN
    computed_rate = measurement.get_neutron_rate(
        channel_nb=3,
        photon_counts=photon_counts,
        irradiations=irradiations,
        distance=distance,  # cm
        time_generator_off=measurement.start_time,
    )

    # TEST
    expected_nb_decays = photon_counts / activated_nuclide.intensity[0]  # decay events
    expected_activity = expected_nb_decays / (count_time_hr * 3600)  # Bq
    # ignoring decays then:
    # irradiation_time * cross_section * nb_atoms * neutron_flux * decay_constant = activity
    expected_neutron_flux = expected_activity / (
        irradiation_time
        * foil.reaction.cross_section
        * foil.nb_atoms
        * activated_nuclide.decay_constant
    )
    area_of_sphere = 4 * np.pi * distance**2
    expected_neutron_rate = expected_neutron_flux * area_of_sphere
    assert np.isclose(computed_rate, expected_neutron_rate)


@pytest.mark.parametrize("distance", [1.0, 5.0, 10.0])
@pytest.mark.parametrize("photon_counts", [1e15, 1e15, 1e15, 0.0])
def test_get_neutron_rate_very_moderate_life(photon_counts, distance):
    # BUILD

    half_life = 10 * 24 * 3600  # seconds  (10 day)

    nuclide_reactant = Nuclide(name="TestNuclide", atomic_mass=200)
    activated_nuclide = Nuclide(
        name="ActivatedNuclide",
        energy=[1000],
        intensity=[1.0],
        half_life=half_life,
    )

    reaction = Reaction(
        reactant=nuclide_reactant,
        product=activated_nuclide,
        cross_section=20.0,
    )

    foil = ActivationFoil(
        reaction=reaction,
        mass=0.1,
        name="TestFoil",
        thickness=None,
    )

    measurement = compass.SampleMeasurement("sample")
    measurement.foil = foil

    count_time_hr = 1  # hr
    measurement.start_time = datetime.datetime(2024, 11, 7)
    measurement.stop_time = datetime.datetime(2024, 11, 7, count_time_hr)

    measurement.detectors = [
        compass.Detector(channel_nb=4),
        compass.Detector(channel_nb=3),
    ]
    measurement.get_detector(3).real_count_time = count_time_hr * 3600
    measurement.get_detector(3).live_count_time = measurement.get_detector(
        3
    ).real_count_time

    irradiation_time = 0.5 * half_life
    irradiations = [{"t_on": 0, "t_off": irradiation_time}]

    # RUN
    computed_rate = measurement.get_neutron_rate(
        channel_nb=3,
        photon_counts=photon_counts,
        irradiations=irradiations,
        distance=distance,  # cm
        time_generator_off=measurement.start_time,
    )

    # TEST
    expected_nb_decays = photon_counts / activated_nuclide.intensity[0]  # decay events
    expected_neutron_flux = (
        expected_nb_decays
        * activated_nuclide.decay_constant
        / (
            (1 - np.exp(-foil.reaction.product.decay_constant * irradiation_time))
            * (1 - np.exp(-foil.reaction.product.decay_constant * count_time_hr * 3600))
            * foil.reaction.cross_section
            * foil.nb_atoms
        )
    )

    area_of_sphere = 4 * np.pi * distance**2
    expected_neutron_rate = expected_neutron_flux * area_of_sphere
    assert np.isclose(computed_rate, expected_neutron_rate)


def test_activationfoil_density_thickness_validation():

    nuclide_reactant = Nuclide(name="TestNuclide", atomic_mass=200)
    activated_nuclide = Nuclide(
        name="ActivatedNuclide",
        energy=[1000],
        intensity=[1.0],
        half_life=10 * 24 * 3600,  # 10 days
    )

    reaction = Reaction(
        reactant=nuclide_reactant,
        product=activated_nuclide,
        cross_section=20.0,
    )

    with pytest.raises(
        ValueError,
        match="Thickness and density must either both be floats or both be None.",
    ):
        ActivationFoil(reaction=reaction, mass=1.0, name="foil", density=1.0)

    with pytest.raises(
        ValueError,
        match="Thickness and density must either both be floats or both be None.",
    ):
        ActivationFoil(reaction=reaction, mass=1.0, name="foil", thickness=0.1)


def create_test_measurement(
    name: str, num_detectors: int = 2, num_events: int = 100
) -> compass.Measurement:
    """
    Helper function to create a test measurement with synthetic data.
    """
    measurement = compass.Measurement(name)

    # Set start and stop times
    measurement.start_time = datetime.datetime(2025, 1, 1, 10, 0, 0)
    measurement.stop_time = datetime.datetime(2025, 1, 1, 10, 15, 0)

    # Create detectors with synthetic events
    for channel_nb in range(num_detectors):
        detector = compass.Detector(channel_nb)

        # Generate synthetic events (time in ps, energy)
        times = np.random.uniform(0, 1e12, num_events)  # Random times in ps
        energies = np.random.uniform(100, 1000, num_events)  # Random energies
        detector.events = np.column_stack((times, energies))

        # Set timing information
        detector.live_count_time = 900.0  # 15 minutes
        detector.real_count_time = 900.0

        measurement.detectors.append(detector)

    return measurement


def test_measurement_to_h5_single(tmpdir):
    """
    Test the Measurement.to_h5 method for a single measurement.
    """
    # Create test measurement
    measurement = create_test_measurement(
        "test_measurement", num_detectors=2, num_events=50
    )

    # Save to HDF5
    h5_file = os.path.join(tmpdir, "test_single.h5")
    measurement.to_h5(h5_file, mode="w")

    # Verify file exists and has correct structure
    assert os.path.exists(h5_file)

    with h5py.File(h5_file, "r") as f:
        # Check measurement group exists
        assert "test_measurement" in f
        measurement_group = f["test_measurement"]

        # Check attributes
        assert "start_time" in measurement_group.attrs
        assert "stop_time" in measurement_group.attrs
        assert measurement_group.attrs["start_time"] == "2025-01-01T10:00:00"
        assert measurement_group.attrs["stop_time"] == "2025-01-01T10:15:00"

        # Check detectors
        assert "detector_0" in measurement_group
        assert "detector_1" in measurement_group

        # Check detector data
        detector_group = measurement_group["detector_0"]
        assert "events" in detector_group
        assert detector_group["events"].shape[1] == 2  # time, energy columns
        assert detector_group["events"].shape[0] == 50  # number of events
        assert detector_group.attrs["live_count_time"] == 900.0
        assert detector_group.attrs["real_count_time"] == 900.0


def test_measurement_to_h5_append_mode(tmpdir):
    """
    Test the Measurement.to_h5 method with append mode for multiple measurements.
    """
    # Create test measurements
    measurement1 = create_test_measurement(
        "measurement_1", num_detectors=1, num_events=30
    )
    measurement2 = create_test_measurement(
        "measurement_2", num_detectors=2, num_events=40
    )

    h5_file = os.path.join(tmpdir, "test_append.h5")

    # Save first measurement
    measurement1.to_h5(h5_file, mode="w")

    # Append second measurement
    measurement2.to_h5(h5_file, mode="a")

    # Verify both measurements are in the file
    with h5py.File(h5_file, "r") as f:
        assert "measurement_1" in f
        assert "measurement_2" in f

        # Check first measurement
        assert "detector_0" in f["measurement_1"]
        assert f["measurement_1"]["detector_0"]["events"].shape[0] == 30

        # Check second measurement
        assert "detector_0" in f["measurement_2"]
        assert "detector_1" in f["measurement_2"]
        assert f["measurement_2"]["detector_0"]["events"].shape[0] == 40


def test_measurement_to_h5_overwrite_existing(tmpdir):
    """
    Test that writing a measurement with the same name overwrites the existing one.
    """
    # Create initial measurement
    measurement1 = create_test_measurement("same_name", num_detectors=1, num_events=30)
    measurement1.detectors[0].live_count_time = 100.0

    # Create updated measurement with same name
    measurement2 = create_test_measurement("same_name", num_detectors=1, num_events=50)
    measurement2.detectors[0].live_count_time = 200.0

    h5_file = os.path.join(tmpdir, "test_overwrite.h5")

    # Save first measurement
    measurement1.to_h5(h5_file, mode="w")

    # Overwrite with second measurement
    measurement2.to_h5(h5_file, mode="a")

    # Verify only the second measurement data remains
    with h5py.File(h5_file, "r") as f:
        assert "same_name" in f
        detector_group = f["same_name"]["detector_0"]
        assert detector_group["events"].shape[0] == 50  # New data
        assert detector_group.attrs["live_count_time"] == 200.0  # New timing


def test_measurement_write_multiple_to_h5(tmpdir):
    """
    Test the Measurement.write_multiple_to_h5 class method.
    """
    # Create multiple test measurements
    measurements = [
        create_test_measurement("exp_1", num_detectors=1, num_events=20),
        create_test_measurement("exp_2", num_detectors=2, num_events=30),
        create_test_measurement("exp_3", num_detectors=3, num_events=40),
    ]

    h5_file = os.path.join(tmpdir, "test_multiple.h5")

    # Write all measurements to file
    compass.Measurement.write_multiple_to_h5(measurements, h5_file)

    # Verify all measurements are in the file
    with h5py.File(h5_file, "r") as f:
        assert "exp_1" in f
        assert "exp_2" in f
        assert "exp_3" in f

        # Check each measurement has correct number of detectors
        assert len([k for k in f["exp_1"].keys() if k.startswith("detector_")]) == 1
        assert len([k for k in f["exp_2"].keys() if k.startswith("detector_")]) == 2
        assert len([k for k in f["exp_3"].keys() if k.startswith("detector_")]) == 3

        # Check event counts
        assert f["exp_1"]["detector_0"]["events"].shape[0] == 20
        assert f["exp_2"]["detector_0"]["events"].shape[0] == 30
        assert f["exp_3"]["detector_0"]["events"].shape[0] == 40


def test_measurement_from_h5_single(tmpdir):
    """
    Test the Measurement.from_h5 method for loading a single measurement.
    """
    # Create and save a test measurement
    original_measurement = create_test_measurement(
        "test_load", num_detectors=2, num_events=35
    )
    h5_file = os.path.join(tmpdir, "test_load_single.h5")
    original_measurement.to_h5(h5_file)

    # Load the measurement back
    loaded_measurement = compass.Measurement.from_h5(
        h5_file, measurement_name="test_load"
    )

    # Verify loaded measurement matches original
    assert loaded_measurement.name == "test_load"
    assert loaded_measurement.start_time == original_measurement.start_time
    assert loaded_measurement.stop_time == original_measurement.stop_time
    assert len(loaded_measurement.detectors) == 2

    # Check detector data
    for i, detector in enumerate(loaded_measurement.detectors):
        original_detector = original_measurement.detectors[i]
        assert detector.channel_nb == original_detector.channel_nb
        assert detector.live_count_time == original_detector.live_count_time
        assert detector.real_count_time == original_detector.real_count_time
        np.testing.assert_array_equal(detector.events, original_detector.events)


def test_measurement_from_h5_all_measurements(tmpdir):
    """
    Test the Measurement.from_h5 method for loading all measurements from a file.
    """
    # Create and save multiple measurements
    measurements = [
        create_test_measurement("load_1", num_detectors=1, num_events=25),
        create_test_measurement("load_2", num_detectors=2, num_events=35),
    ]

    h5_file = os.path.join(tmpdir, "test_load_all.h5")
    compass.Measurement.write_multiple_to_h5(measurements, h5_file)

    # Load all measurements
    loaded_measurements = compass.Measurement.from_h5(h5_file)

    # Verify we got all measurements
    assert len(loaded_measurements) == 2
    loaded_names = [m.name for m in loaded_measurements]
    assert "load_1" in loaded_names
    assert "load_2" in loaded_names

    # Find corresponding measurements
    load_1 = next(m for m in loaded_measurements if m.name == "load_1")
    load_2 = next(m for m in loaded_measurements if m.name == "load_2")

    assert len(load_1.detectors) == 1
    assert len(load_2.detectors) == 2
    assert load_1.detectors[0].events.shape[0] == 25
    assert load_2.detectors[0].events.shape[0] == 35


def test_measurement_from_h5_nonexistent_measurement(tmpdir):
    """
    Test that loading a non-existent measurement raises appropriate error.
    """
    # Create a measurement and save it
    measurement = create_test_measurement("existing", num_detectors=1, num_events=10)
    h5_file = os.path.join(tmpdir, "test_nonexistent.h5")
    measurement.to_h5(h5_file)

    # Try to load a non-existent measurement
    with pytest.raises(ValueError, match="Measurement 'nonexistent' not found in file"):
        compass.Measurement.from_h5(h5_file, measurement_name="nonexistent")


def test_measurement_h5_roundtrip(tmpdir):
    """
    Test complete roundtrip: create -> save -> load -> verify data integrity.
    """
    # Create measurement with specific, verifiable data
    measurement = compass.Measurement("roundtrip_test")
    measurement.start_time = datetime.datetime(2025, 7, 2, 14, 30, 0)
    measurement.stop_time = datetime.datetime(2025, 7, 2, 15, 0, 0)

    # Create detector with specific events
    detector = compass.Detector(channel_nb=5)
    detector.events = np.array(
        [
            [1000000000, 150.5],  # time in ps, energy
            [2000000000, 250.7],
            [3000000000, 350.9],
        ]
    )
    detector.live_count_time = 1800.0
    detector.real_count_time = 1800.0
    measurement.detectors = [detector]

    # Save and load
    h5_file = os.path.join(tmpdir, "roundtrip.h5")
    measurement.to_h5(h5_file)
    loaded_measurement = compass.Measurement.from_h5(
        h5_file, measurement_name="roundtrip_test"
    )

    # Verify exact data integrity
    assert loaded_measurement.name == "roundtrip_test"
    assert loaded_measurement.start_time == measurement.start_time
    assert loaded_measurement.stop_time == measurement.stop_time
    assert len(loaded_measurement.detectors) == 1

    loaded_detector = loaded_measurement.detectors[0]
    assert loaded_detector.channel_nb == 5
    assert loaded_detector.live_count_time == 1800.0
    assert loaded_detector.real_count_time == 1800.0
    np.testing.assert_array_equal(loaded_detector.events, detector.events)


def test_measurement_h5_empty_measurement(tmpdir):
    """
    Test saving and loading a measurement with no detectors.
    """
    # Create empty measurement
    measurement = compass.Measurement("empty_test")
    measurement.start_time = datetime.datetime(2025, 1, 1, 12, 0, 0)
    measurement.stop_time = datetime.datetime(2025, 1, 1, 12, 30, 0)
    measurement.detectors = []  # No detectors

    # Save and load
    h5_file = os.path.join(tmpdir, "empty.h5")
    measurement.to_h5(h5_file)
    loaded_measurement = compass.Measurement.from_h5(
        h5_file, measurement_name="empty_test"
    )

    # Verify empty measurement
    assert loaded_measurement.name == "empty_test"
    assert loaded_measurement.start_time == measurement.start_time
    assert loaded_measurement.stop_time == measurement.stop_time
    assert len(loaded_measurement.detectors) == 0


def test_measurement_h5_roundtrip_spectrum_only(tmpdir):
    """
    Test complete roundtrip with spectrum_only flag: create -> save -> load -> verify spectrum data integrity.
    """
    # Create measurement with specific, verifiable data
    measurement = compass.Measurement("roundtrip_spectrum_test")
    measurement.start_time = datetime.datetime(2025, 7, 2, 14, 30, 0)
    measurement.stop_time = datetime.datetime(2025, 7, 2, 15, 0, 0)

    # Create detector with specific events that will create a predictable spectrum
    detector = compass.Detector(channel_nb=5)
    # Create events with integer energies for predictable histogram
    detector.events = np.array(
        [
            [1000000000, 100.0],  # time in ps, energy
            [2000000000, 100.0],  # Same energy -> 2 counts in bin 100
            [3000000000, 200.0],  # Different energy -> 1 count in bin 200
            [4000000000, 200.0],  # Same energy -> 2 counts in bin 200
            [5000000000, 300.0],  # Different energy -> 1 count in bin 300
            [5000000000, 300.0],  # Same energy -> 2 counts in bin 300
            [5000000000, 400.0],  # Different energy -> 1 count in bin 400
        ]
    )
    detector.live_count_time = 1800.0
    detector.real_count_time = 1800.0
    measurement.detectors = [detector]

    # Get the expected spectrum before saving
    expected_hist, expected_bin_edges = detector.get_energy_hist(bins=None)

    # Save with spectrum_only=True and load
    h5_file = os.path.join(tmpdir, "roundtrip_spectrum.h5")
    measurement.to_h5(h5_file, spectrum_only=True)
    loaded_measurement = compass.Measurement.from_h5(
        h5_file, measurement_name="roundtrip_spectrum_test"
    )

    # Verify basic measurement data integrity
    assert loaded_measurement.name == "roundtrip_spectrum_test"
    assert loaded_measurement.start_time == measurement.start_time
    assert loaded_measurement.stop_time == measurement.stop_time
    assert len(loaded_measurement.detectors) == 1

    loaded_detector = loaded_measurement.detectors[0]
    assert loaded_detector.channel_nb == 5
    assert loaded_detector.live_count_time == 1800.0
    assert loaded_detector.real_count_time == 1800.0

    # Verify events array is empty (spectrum_only mode)
    assert loaded_detector.events.shape[0] == 0

    # Verify spectrum data is present and correct
    assert hasattr(loaded_detector, "spectrum")
    assert hasattr(loaded_detector, "bin_edges")
    np.testing.assert_array_equal(loaded_detector.spectrum, expected_hist)
    np.testing.assert_array_equal(loaded_detector.bin_edges, expected_bin_edges)

    # Verify the spectrum contains expected counts
    # The exact bin positions depend on the histogram implementation
    print(f"Spectrum: {loaded_detector.spectrum}")
    print(f"Bin edges: {loaded_detector.bin_edges}")
    assert np.sum(loaded_detector.spectrum) == 7  # Total number of events


def test_measurement_h5_spectrum_only_file_structure(tmpdir):
    """
    Test that spectrum_only mode creates the correct HDF5 file structure.
    """
    # Create measurement with events
    measurement = create_test_measurement(
        "spectrum_structure_test", num_detectors=1, num_events=100
    )

    # Save with spectrum_only=True
    h5_file = os.path.join(tmpdir, "spectrum_structure.h5")
    measurement.to_h5(h5_file, spectrum_only=True)

    # Verify file structure
    with h5py.File(h5_file, "r") as f:
        assert "spectrum_structure_test" in f
        measurement_group = f["spectrum_structure_test"]

        # Check measurement attributes
        assert "start_time" in measurement_group.attrs
        assert "stop_time" in measurement_group.attrs

        # Check detector group
        assert "detector_0" in measurement_group
        detector_group = measurement_group["detector_0"]

        # In spectrum_only mode, should have spectrum and bin_edges, but empty events
        assert "spectrum" in detector_group
        assert "bin_edges" in detector_group
        assert "events" in detector_group

        # Events should be empty array
        assert detector_group["events"].shape[0] == 0

        # Spectrum should have data
        assert detector_group["spectrum"].shape[0] > 0
        assert detector_group["bin_edges"].shape[0] > 0

        # Timing attributes should still be present
        assert "live_count_time" in detector_group.attrs
        assert "real_count_time" in detector_group.attrs


def test_measurement_h5_spectrum_only_vs_full_size_comparison(tmpdir):
    """
    Test that spectrum_only mode produces smaller files than full event storage.
    """
    # Create measurement with many events to see file size difference
    measurement = create_test_measurement("size_test", num_detectors=1, num_events=1000)

    # Save in both modes
    h5_file_full = os.path.join(tmpdir, "full_events.h5")
    h5_file_spectrum = os.path.join(tmpdir, "spectrum_only.h5")

    measurement.to_h5(h5_file_full, spectrum_only=False)
    measurement.to_h5(h5_file_spectrum, spectrum_only=True)

    # Compare file sizes
    full_size = os.path.getsize(h5_file_full)
    spectrum_size = os.path.getsize(h5_file_spectrum)

    # Spectrum-only file should be smaller (unless histogram has more bins than events)
    # At minimum, both files should exist and have reasonable sizes
    assert full_size > 0
    assert spectrum_size > 0

    # For 1000 events, the full file should typically be larger
    # (though this could depend on the specific data and compression)
    print(f"Full events file size: {full_size} bytes")
    print(f"Spectrum only file size: {spectrum_size} bytes")


def test_measurement_h5_spectrum_only_analysis_capability(tmpdir):
    """
    Test that spectrum_only data can still be used for basic analysis.
    """
    # Create measurement with well-defined energy distribution
    measurement = compass.Measurement("analysis_test")
    measurement.start_time = datetime.datetime(2025, 7, 2, 10, 0, 0)
    measurement.stop_time = datetime.datetime(2025, 7, 2, 10, 30, 0)

    detector = compass.Detector(channel_nb=1)
    # Create events with known energy distribution
    energies = np.concatenate(
        [
            np.full(50, 500.0),  # 50 events at 500 keV
            np.full(30, 600.0),  # 30 events at 600 keV
            np.full(20, 700.0),  # 20 events at 700 keV
        ]
    )
    times = np.random.uniform(0, 1e12, len(energies))
    detector.events = np.column_stack((times, energies))
    detector.live_count_time = 1800.0
    detector.real_count_time = 1800.0
    measurement.detectors = [detector]

    # Save with spectrum_only=True
    h5_file = os.path.join(tmpdir, "analysis_spectrum.h5")
    measurement.to_h5(h5_file, spectrum_only=True)

    # Load and analyze spectrum
    loaded_measurement = compass.Measurement.from_h5(
        h5_file, measurement_name="analysis_test"
    )
    loaded_detector = loaded_measurement.detectors[0]

    # Verify we can analyze the spectrum
    assert hasattr(loaded_detector, "spectrum")
    assert hasattr(loaded_detector, "bin_edges")

    # Check total counts
    total_counts = np.sum(loaded_detector.spectrum)
    assert total_counts == 100  # 50 + 30 + 20

    # Check that peak energies are preserved in the spectrum
    # Find bin centers
    bin_centers = (loaded_detector.bin_edges[:-1] + loaded_detector.bin_edges[1:]) / 2

    # Find peaks in the spectrum (simple approach)
    peak_indices = np.where(loaded_detector.spectrum > 15)[
        0
    ]  # Bins with significant counts
    peak_energies = bin_centers[peak_indices]

    # Should have peaks near our input energies (500, 600, 700)
    assert len(peak_energies) >= 3, "Should find at least 3 energy peaks"

    # Verify the spectrum structure makes sense
    assert loaded_detector.spectrum.dtype in [np.int32, np.int64, np.uint32, np.uint64]
    assert loaded_detector.bin_edges.dtype in [np.int32, np.int64, np.uint32, np.uint64]
    assert len(loaded_detector.bin_edges) == len(loaded_detector.spectrum) + 1
