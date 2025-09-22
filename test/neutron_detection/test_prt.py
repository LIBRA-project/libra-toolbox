import h5py
import numpy as np
import pytest

from libra_toolbox.neutron_detection.diamond import prt

ns_to_s = 1e-9


@pytest.fixture
def h5_file_with_data(tmpdir):
    """Fixture to create an HDF5 file with structured data for testing."""
    filename = tmpdir + "/test_file.h5"

    # Create structured array for Channel A
    data_channel_a = np.zeros(
        100, dtype=[("Time [ns]", float), ("Amplitude [mV]", float)]
    )
    data_channel_a["Time [ns]"] = np.random.rand(100) * 1e9  # Example time values in ns
    data_channel_a["Amplitude [mV]"] = (
        np.random.rand(100) * 100
    )  # Example amplitude values in mV

    # Create structured array for Channel B
    data_channel_b = np.zeros(
        100, dtype=[("Time [ns]", float), ("Amplitude [mV]", float)]
    )
    data_channel_b["Time [ns]"] = np.random.rand(100) * 1e9
    data_channel_b["Amplitude [mV]"] = np.random.rand(100) * 100

    # Create HDF5 file
    with h5py.File(filename, "w") as f:
        f.attrs["Active channels"] = [True, True]

        channel_a = f.create_group("Channel A")
        channel_a.create_dataset(name="Amplitude-Timestamp", data=data_channel_a)

        channel_b = f.create_group("Channel B")
        channel_b.create_dataset(name="Amplitude-Timestamp", data=data_channel_b)

    return filename, data_channel_a, data_channel_b


def test_get_timestamps_and_amplitudes(h5_file_with_data):
    """
    Test the get_timestamps_and_amplitudes function.
    This function retrieves timestamps and amplitudes from a given HDF5 file.
    It checks if the retrieved data matches the expected data.

    Args:
        h5_file_with_data: Fixture that provides a temporary HDF5 file with structured data.
    """
    filename, data_channel_a, _ = h5_file_with_data

    # run
    with h5py.File(filename, "r") as ROSY_file:
        timestamps, amplitudes = prt.get_timestamps_and_amplitudes(
            ROSY_file, channel="Channel A"
        )

    # test
    assert np.array_equal(timestamps, data_channel_a["Time [ns]"] * ns_to_s)
    assert np.array_equal(amplitudes, data_channel_a["Amplitude [mV]"])


def test_load_data_from_file(h5_file_with_data):
    """
    Test the load_data_from_file function.
    This function loads data from a given HDF5 file and checks if the loaded data
    matches the expected data.

    Args:
        h5_file_with_data: Fixture that provides a temporary HDF5 file with structured data.
    """
    filename, data_channel_a, data_channel_b = h5_file_with_data

    data = prt.load_data_from_file(filename)
    assert "Channel A" in data
    assert "Channel B" in data
    assert np.array_equal(
        data["Channel A"]["timestamps"],
        data_channel_a["Time [ns]"] * ns_to_s,
    )
    assert np.array_equal(
        data["Channel A"]["amplitudes"], data_channel_a["Amplitude [mV]"]
    )
    assert np.array_equal(
        data["Channel B"]["timestamps"],
        data_channel_b["Time [ns]"] * ns_to_s,
    )
    assert np.array_equal(
        data["Channel B"]["amplitudes"], data_channel_b["Amplitude [mV]"]
    )


@pytest.mark.parametrize("bin_time", [1, 10, 100])
@pytest.mark.parametrize("count_rate_real", [1, 10, 100])
def test_get_count_rate(bin_time: float, count_rate_real: float):
    """
    Test the get_count_rate function.
    This function calculates the count rate from given timestamps and checks
    if the calculated count rate matches the expected count rate.
    Args:
        bin_time: The bin time in seconds.
        count_rate_real: The expected count rate in Hz.
    """
    # Example data
    total_time = 1000  # seconds
    timestamps = np.linspace(0, total_time, num=count_rate_real * total_time)

    # run
    count_rates, _ = prt.get_count_rate(timestamps, bin_time=bin_time)

    # test
    assert np.allclose(count_rates, count_rate_real)


@pytest.mark.parametrize(
    "ch1_times, ch2_times, ch1_ampl, ch2_ampl, t_window, expected",
    [
        # Test case 1: Simple match within time window
        (
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [10, 20, 30],
            [15, 25, 35],
            0.2,
            (
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 3.1],
                [10, 20, 30],
                [15, 25, 35],
            ),
        ),
        # Test case 2: No match due to time window
        (
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [10, 20, 30],
            [15, 25, 35],
            0.1,
            ([], [], [], []),
        ),
        # Test case 3: Partial match
        (
            [1.0, 2.0, 3.0],
            [1.05, 3.05, 5.0],
            [10, 20, 30],
            [15, 35, 25],
            0.1,
            (
                [1.0, 3.0],
                [1.05, 3.05],
                [10, 30],
                [15, 35],
            ),
        ),
        # Test case 4: Empty input
        (
            [],
            [],
            [],
            [],
            0.1,
            ([], [], [], []),
        ),
    ],
)
def test_coinc_2(ch1_times, ch2_times, ch1_ampl, ch2_ampl, t_window, expected):
    """
    Test the coinc_2 function.
    This function checks if the coincidence detection works correctly
    for two channels within a given time window.

    Args:
        ch1_times: List of timestamps for channel 1.
        ch2_times: List of timestamps for channel 2.
        ch1_ampl: List of amplitudes for channel 1.
        ch2_ampl: List of amplitudes for channel 2.
        t_window: Time window for coincidence detection.
        expected: Expected output (time and amplitude matches).
    """
    result = prt.coinc_2(ch1_times, ch2_times, ch1_ampl, ch2_ampl, t_window)
    # convert everything to list
    result = (
        result[0].tolist(),
        result[1].tolist(),
        result[2].tolist(),
        result[3].tolist(),
    )
    assert result == expected


@pytest.mark.parametrize(
    "ch1_times, ch2_times, ch3_times, ch1_ampl, ch2_ampl, ch3_ampl, t_window, expected",
    [
        # Test case 1: All channels match within the time window
        (
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [1.15, 2.15, 3.15],
            [10, 20, 30],
            [15, 25, 35],
            [12, 22, 32],
            0.2,
            (
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 3.1],
                [1.15, 2.15, 3.15],
                [10, 20, 30],
                [15, 25, 35],
                [12, 22, 32],
            ),
        ),
        # Test case 2: No matches due to time window
        (
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10, 20, 30],
            [15, 25, 35],
            [12, 22, 32],
            0.1,
            ([], [], [], [], [], []),
        ),
        # Test case 3: Partial matches
        (
            [1.0, 2.0, 3.0],
            [1.05, 1.8, 3.05],
            [1.1, 2.1, 3.1],
            [10, 20, 30],
            [15, 25, 35],
            [12, 22, 32],
            0.11,
            (
                [1.0, 3.0],
                [1.05, 3.05],
                [1.1, 3.1],
                [10, 30],
                [15, 35],
                [12, 32],
            ),
        ),
        # Test case 4: Empty input
        (
            [],
            [],
            [],
            [],
            [],
            [],
            0.1,
            ([], [], [], [], [], []),
        ),
    ],
)
def test_coinc_3(
    ch1_times, ch2_times, ch3_times, ch1_ampl, ch2_ampl, ch3_ampl, t_window, expected
):
    """
    Test the coinc_3 function.
    This function checks if the coincidence detection works correctly
    for three channels within a given time window.

    Args:
        ch1_times: List of timestamps for channel 1.
        ch2_times: List of timestamps for channel 2.
        ch3_times: List of timestamps for channel 3.
        ch1_ampl: List of amplitudes for channel 1.
        ch2_ampl: List of amplitudes for channel 2.
        ch3_ampl: List of amplitudes for channel 3.
        t_window: Time window for coincidence detection.
        expected: Expected output (time and amplitude matches).
    """
    result = prt.coinc_3(
        ch1_times, ch2_times, ch3_times, ch1_ampl, ch2_ampl, ch3_ampl, t_window
    )
    # convert everything to list
    result = (
        result[0].tolist(),
        result[1].tolist(),
        result[2].tolist(),
        result[3].tolist(),
        result[4].tolist(),
        result[5].tolist(),
    )
    assert result == expected


@pytest.mark.parametrize(
    "ch1_times, ch2_times, ch3_times, ch4_times, ch1_ampl, ch2_ampl, ch3_ampl, ch4_ampl, t_window, expected",
    [
        # Test case 1: All channels match within the time window
        (
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 3.1],
            [1.15, 2.15, 3.15],
            [1.2, 2.2, 3.2],
            [10, 20, 30],
            [15, 25, 35],
            [12, 22, 32],
            [14, 24, 34],
            0.3,
            (
                [1.0, 2.0, 3.0],
                [1.1, 2.1, 3.1],
                [1.15, 2.15, 3.15],
                [1.2, 2.2, 3.2],
                [10, 20, 30],
                [15, 25, 35],
                [12, 22, 32],
                [14, 24, 34],
            ),
        ),
        # Test case 2: No matches due to time window
        (
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
            [10, 20, 30],
            [15, 25, 35],
            [12, 22, 32],
            [14, 24, 34],
            0.1,
            ([], [], [], [], [], [], [], []),
        ),
        # Test case 3: Partial matches
        (
            [1.0, 2.0, 3.0],
            [1.05, 2.05, 5.0],
            [1.1, 2.1, 3.1],
            [1.15, 2.15, 3.15],
            [10, 20, 30],
            [15, 25, 35],
            [12, 22, 32],
            [14, 24, 34],
            0.2,
            (
                [1.0, 2.0],
                [1.05, 2.05],
                [1.1, 2.1],
                [1.15, 2.15],
                [10, 20],
                [15, 25],
                [12, 22],
                [14, 24],
            ),
        ),
        # Test case 4: Empty input
        (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            0.1,
            ([], [], [], [], [], [], [], []),
        ),
    ],
)
def test_coinc_4(
    ch1_times,
    ch2_times,
    ch3_times,
    ch4_times,
    ch1_ampl,
    ch2_ampl,
    ch3_ampl,
    ch4_ampl,
    t_window,
    expected,
):
    """
    Test the coinc_4 function.
    This function checks if the coincidence detection works correctly
    for four channels within a given time window.

    Args:
        ch1_times: List of timestamps for channel 1.
        ch2_times: List of timestamps for channel 2.
        ch3_times: List of timestamps for channel 3.
        ch4_times: List of timestamps for channel 4.
        ch1_ampl: List of amplitudes for channel 1.
        ch2_ampl: List of amplitudes for channel 2.
        ch3_ampl: List of amplitudes for channel 3.
        ch4_ampl: List of amplitudes for channel 4.
        t_window: Time window for coincidence detection.
        expected: Expected output (time and amplitude matches).
    """
    result = prt.coinc_4(
        ch1_times,
        ch2_times,
        ch3_times,
        ch4_times,
        ch1_ampl,
        ch2_ampl,
        ch3_ampl,
        ch4_ampl,
        t_window,
    )
    # convert everything to list
    result = (
        result[0].tolist(),
        result[1].tolist(),
        result[2].tolist(),
        result[3].tolist(),
        result[4].tolist(),
        result[5].tolist(),
        result[6].tolist(),
        result[7].tolist(),
    )
    assert result == expected


@pytest.mark.parametrize(
    "ch1_times, ch2_times, ch3_times, ch1_ampl, ch2_ampl, t_window, expected",
    [
        # Test case 1: Coincidence between Ch1 and Ch2, no events in Ch3
        (
            [1.0, 2.0, 3.0],  # ch1_times
            [1.1, 2.1, 3.1],  # ch2_times
            [4.0, 5.0, 6.0],  # ch3_times (no overlap)
            [10, 20, 30],  # ch1_ampl
            [15, 25, 35],  # ch2_ampl
            0.2,  # t_window
            (
                [1.0, 2.0, 3.0],  # ch1_times (coincidence)
                [1.1, 2.1, 3.1],  # ch2_times (coincidence)
                [10, 20, 30],  # ch1_ampl
                [15, 25, 35],  # ch2_ampl
            ),
        ),
        # Test case 2: Coincidence between Ch1 and Ch2, but Ch3 overlaps for first event
        (
            [1.0, 2.0, 3.0],  # ch1_times
            [1.1, 2.1, 3.1],  # ch2_times
            [1.05, 4, 5],  # ch3_times
            [10, 20, 30],  # ch1_ampl
            [15, 25, 35],  # ch2_ampl
            0.2,  # t_window
            (
                [2.0, 3.0],  # ch1_times (coincidence)
                [2.1, 3.1],  # ch2_times (coincidence)
                [20, 30],  # ch1_ampl
                [25, 35],  # ch2_ampl
            ),
        ),
        # Test case 3: No matches between Ch1 and Ch2
        (
            [1.0, 2.0, 3.0],  # ch1_times
            [4.0, 5.0, 6.0],  # ch2_times (no overlap)
            [7.0, 8.0, 9.0],  # ch3_times
            [10, 20, 30],  # ch1_ampl
            [15, 25, 35],  # ch2_ampl
            0.2,  # t_window
            (
                [],  # No valid ch1_times
                [],  # No valid ch2_times
                [],  # No valid ch1_ampl
                [],  # No valid ch2_ampl
            ),
        ),
        # Test case 4: Empty input
        (
            [],  # ch1_times
            [],  # ch2_times
            [],  # ch3_times
            [],  # ch1_ampl
            [],  # ch2_ampl
            0.2,  # t_window
            (
                [],  # No valid ch1_times
                [],  # No valid ch2_times
                [],  # No valid ch1_ampl
                [],  # No valid ch2_ampl
            ),
        ),
    ],
)
def test_coinc_2_anti_1(
    ch1_times, ch2_times, ch3_times, ch1_ampl, ch2_ampl, t_window, expected
):
    """
    Test the coinc_2_anti_1 function.
    This function checks if the coincidence detection works correctly
    for two channels with anti-coincidence on a third channel.

    Args:
        ch1_times: List of timestamps for channel 1.
        ch2_times: List of timestamps for channel 2.
        ch3_times: List of timestamps for channel 3 (anti-coincidence).
        ch1_ampl: List of amplitudes for channel 1.
        ch2_ampl: List of amplitudes for channel 2.
        t_window: Time window for coincidence detection.
        expected: Expected output (time and amplitude matches).
    """
    result = prt.coinc_2_anti_1(
        ch1_times, ch2_times, ch3_times, ch1_ampl, ch2_ampl, t_window
    )

    # Convert result to lists for comparison
    result = tuple(r.tolist() for r in result)

    assert result == expected


@pytest.mark.parametrize(
    "ch1_times, ch2_times, ch3_times, ch4_times, ch1_ampl, ch2_ampl, ch3_ampl, t_window, expected",
    [
        # Test case 1: Coincidence between Ch1, Ch2, and Ch3, no events in Ch4
        (
            [1.0, 2.0, 3.0],  # ch1_times
            [1.1, 2.1, 3.1],  # ch2_times
            [1.05, 2.05, 3.05],  # ch3_times
            [4.0, 5.0, 6.0],  # ch4_times (no overlap)
            [10, 20, 30],  # ch1_ampl
            [15, 25, 35],  # ch2_ampl
            [12, 22, 32],  # ch3_ampl
            0.2,  # t_window
            (
                [1.0, 2.0, 3.0],  # ch1_times
                [1.1, 2.1, 3.1],  # ch2_times
                [1.05, 2.05, 3.05],  # ch3_times
                [10, 20, 30],  # ch1_ampl
                [15, 25, 35],  # ch2_ampl
                [12, 22, 32],  # ch3_ampl
            ),
        ),
        # Test case 2: Coincidence between Ch1, Ch2, and Ch3, but Ch4 overlaps
        (
            [1.0, 2.0, 3.0],  # ch1_times
            [1.1, 2.1, 3.1],  # ch2_times
            [1.05, 2.05, 3.05],  # ch3_times
            [1.02, 4, 5],  # ch4_times (overlaps with Ch1, Ch2, and Ch3)
            [10, 20, 30],  # ch1_ampl
            [15, 25, 35],  # ch2_ampl
            [12, 22, 32],  # ch3_ampl
            0.2,  # t_window
            (
                [2.0, 3.0],  # ch1_times
                [2.1, 3.1],  # ch2_times
                [2.05, 3.05],  # ch3_times
                [20, 30],  # ch1_ampl
                [25, 35],  # ch2_ampl
                [22, 32],  # ch3_ampl
            ),
        ),
        # Test case 3: No matches between Ch1, Ch2, and Ch3
        (
            [1.0, 2.0, 3.0],  # ch1_times
            [4.0, 5.0, 6.0],  # ch2_times (no overlap)
            [7.0, 8.0, 9.0],  # ch3_times
            [10.0, 11.0, 12.0],  # ch4_times
            [10, 20, 30],  # ch1_ampl
            [15, 25, 35],  # ch2_ampl
            [12, 22, 32],  # ch3_ampl
            0.2,  # t_window
            (
                [],  # No valid ch1_times
                [],  # No valid ch2_times
                [],  # No valid ch3_times
                [],  # No valid ch1_ampl
                [],  # No valid ch2_ampl
                [],  # No valid ch3_ampl
            ),
        ),
        # Test case 4: Empty input
        (
            [],  # ch1_times
            [],  # ch2_times
            [],  # ch3_times
            [],  # ch4_times
            [],  # ch1_ampl
            [],  # ch2_ampl
            [],  # ch3_ampl
            0.2,  # t_window
            (
                [],  # No valid ch1_times
                [],  # No valid ch2_times
                [],  # No valid ch3_times
                [],  # No valid ch1_ampl
                [],  # No valid ch2_ampl
                [],  # No valid ch3_ampl
            ),
        ),
    ],
)
def test_coinc_3_anti_1(
    ch1_times,
    ch2_times,
    ch3_times,
    ch4_times,
    ch1_ampl,
    ch2_ampl,
    ch3_ampl,
    t_window,
    expected,
):
    """
    Test the coinc_3_anti_1 function.
    This function checks if the coincidence detection works correctly
    for three channels with anti-coincidence on a fourth channel.

    Args:
        ch1_times: List of timestamps for channel 1.
        ch2_times: List of timestamps for channel 2.
        ch3_times: List of timestamps for channel 3.
        ch4_times: List of timestamps for channel 4 (anti-coincidence).
        ch1_ampl: List of amplitudes for channel 1.
        ch2_ampl: List of amplitudes for channel 2.
        ch3_ampl: List of amplitudes for channel 3.
        t_window: Time window for coincidence detection.
        expected: Expected output (time and amplitude matches).
    """
    result = prt.coinc_3_anti_1(
        ch1_times,
        ch2_times,
        ch3_times,
        ch4_times,
        ch1_ampl,
        ch2_ampl,
        ch3_ampl,
        t_window,
    )

    # Convert result to lists for comparison
    result = tuple(r.tolist() for r in result)

    assert result == expected


@pytest.mark.parametrize(
    "ch1_times, ch2_times, ch3_times, ch4_times, ch1_ampl, ch2_ampl, t_window, expected",
    [
        # Test case 1: Coincidence between Ch1 and Ch2, no events in Ch3 and Ch4
        (
            [1.0, 2.0, 3.0],  # ch1_times
            [1.1, 2.1, 3.1],  # ch2_times
            [4.0, 5.0, 6.0],  # ch3_times (no overlap)
            [7.0, 8.0, 9.0],  # ch4_times (no overlap)
            [10, 20, 30],  # ch1_ampl
            [15, 25, 35],  # ch2_ampl
            0.2,  # t_window
            (
                [1.0, 2.0, 3.0],  # ch1_times
                [1.1, 2.1, 3.1],  # ch2_times
                [10, 20, 30],  # ch1_ampl
                [15, 25, 35],  # ch2_ampl
            ),
        ),
        # Test case 2: Coincidence between Ch1 and Ch2, but Ch3 overlaps
        (
            [1.0, 2.0, 3.0],  # ch1_times
            [1.1, 2.1, 3.1],  # ch2_times
            [1.05, 2.05, 5.0],  # ch3_times (overlaps with Ch1 and Ch2)
            [7.0, 8.0, 9.0],  # ch4_times (no overlap)
            [10, 20, 30],  # ch1_ampl
            [15, 25, 35],  # ch2_ampl
            0.2,  # t_window
            (
                [3.0],  # ch1_times
                [3.1],  # ch2_times
                [30],  # ch1_ampl
                [35],  # ch2_ampl
            ),
        ),
        # Test case 3: Coincidence between Ch1 and Ch2, but Ch4 overlaps
        (
            [1.0, 2.0, 3.0],  # ch1_times
            [1.1, 2.1, 3.1],  # ch2_times
            [4.0, 5.0, 6.0],  # ch3_times (no overlap)
            [1.05, 2.05, 8.0],  # ch4_times (overlaps with Ch1 and Ch2)
            [10, 20, 30],  # ch1_ampl
            [15, 25, 35],  # ch2_ampl
            0.2,  # t_window
            (
                [3.0],  # ch1_times
                [3.1],  # ch2_times
                [30],  # ch1_ampl
                [35],  # ch2_ampl
            ),
        ),
        # Test case 4: No matches between Ch1 and Ch2
        (
            [1.0, 2.0, 3.0],  # ch1_times
            [4.0, 5.0, 6.0],  # ch2_times (no overlap)
            [7.0, 8.0, 9.0],  # ch3_times
            [10.0, 11.0, 12.0],  # ch4_times
            [10, 20, 30],  # ch1_ampl
            [15, 25, 35],  # ch2_ampl
            0.2,  # t_window
            (
                [],  # No valid ch1_times
                [],  # No valid ch2_times
                [],  # No valid ch1_ampl
                [],  # No valid ch2_ampl
            ),
        ),
        # Test case 5: Empty input
        (
            [],  # ch1_times
            [],  # ch2_times
            [],  # ch3_times
            [],  # ch4_times
            [],  # ch1_ampl
            [],  # ch2_ampl
            0.2,  # t_window
            (
                [],  # No valid ch1_times
                [],  # No valid ch2_times
                [],  # No valid ch1_ampl
                [],  # No valid ch2_ampl
            ),
        ),
    ],
)
def test_coinc_2_anti_2(
    ch1_times,
    ch2_times,
    ch3_times,
    ch4_times,
    ch1_ampl,
    ch2_ampl,
    t_window,
    expected,
):
    """
    Test the coinc_2_anti_2 function.
    This function checks if the coincidence detection works correctly
    for two channels with anti-coincidence on two other channels.

    Args:
        ch1_times: List of timestamps for channel 1.
        ch2_times: List of timestamps for channel 2.
        ch3_times: List of timestamps for channel 3 (anti-coincidence).
        ch4_times: List of timestamps for channel 4 (anti-coincidence).
        ch1_ampl: List of amplitudes for channel 1.
        ch2_ampl: List of amplitudes for channel 2.
        t_window: Time window for coincidence detection.
        expected: Expected output (time and amplitude matches).
    """
    result = prt.coinc_2_anti_2(
        ch1_times,
        ch2_times,
        ch3_times,
        ch4_times,
        ch1_ampl,
        ch2_ampl,
        t_window,
    )

    # Convert result to lists for comparison
    result = tuple(r.tolist() for r in result)

    assert result == expected


@pytest.mark.parametrize(
    "A_time, A_ampl, B_time, B_ampl, C_time, C_ampl, D_time, D_ampl, coincidence_window, coincidence_citeria, expected",
    [
        # Test case 1: Coincidence between A and B only
        (
            [1.0, 2.0, 3.0],  # A_time
            [10, 20, 30],  # A_ampl
            [1.1, 2.1, 3.1],  # B_time
            [15, 25, 35],  # B_ampl
            [],  # C_time
            [],  # C_ampl
            [],  # D_time
            [],  # D_ampl
            0.2,  # coincidence_window
            [1, 1, 0, 0],  # coincidence_citeria
            {
                "A_time [s]": [1.0, 2.0, 3.0],
                "A_amplitude [mV]": [10, 20, 30],
                "B_time [s]": [1.1, 2.1, 3.1],
                "B_amplitude [mV]": [15, 25, 35],
                "Sum_amplitude [mV]": [25, 45, 65],
            },
        ),
        # Test case 2: Coincidence between A, B, and C
        (
            [1.0, 2.0, 3.0],  # A_time
            [10, 20, 30],  # A_ampl
            [1.1, 2.1, 3.1],  # B_time
            [15, 25, 35],  # B_ampl
            [1.05, 2.05, 3.05],  # C_time
            [12, 22, 32],  # C_ampl
            [],  # D_time
            [],  # D_ampl
            0.2,  # coincidence_window
            [1, 1, 1, 0],  # coincidence_citeria
            {
                "A_time [s]": [1.0, 2.0, 3.0],
                "A_amplitude [mV]": [10, 20, 30],
                "B_time [s]": [1.1, 2.1, 3.1],
                "B_amplitude [mV]": [15, 25, 35],
                "C_time [s]": [1.05, 2.05, 3.05],
                "C_amplitude [mV]": [12, 22, 32],
                "Sum_amplitude [mV]": [37, 67, 97],
            },
        ),
        # Test case 3: Coincidence between A and B with anti-coincidence on C
        (
            [1.0, 2.0, 3.0],  # A_time
            [10, 20, 30],  # A_ampl
            [1.1, 2.1, 3.1],  # B_time
            [15, 25, 35],  # B_ampl
            [1.05, 4, 5],  # C_time
            [12, 22, 32],  # C_ampl
            [],  # D_time
            [],  # D_ampl
            0.2,  # coincidence_window
            [1, 1, 2, 0],  # coincidence_citeria
            {
                "A_time [s]": [2.0, 3.0],
                "A_amplitude [mV]": [20, 30],
                "B_time [s]": [2.1, 3.1],
                "B_amplitude [mV]": [25, 35],
            },
        ),
        # Test case 4: No matches due to time window
        (
            [1.0, 2.0, 3.0],  # A_time
            [10, 20, 30],  # A_ampl
            [4.0, 5.0, 6.0],  # B_time
            [15, 25, 35],  # B_ampl
            [],  # C_time
            [],  # C_ampl
            [],  # D_time
            [],  # D_ampl
            0.1,  # coincidence_window
            [1, 1, 0, 0],  # coincidence_citeria
            {
                "A_time [s]": [],
                "A_amplitude [mV]": [],
                "B_time [s]": [],
                "B_amplitude [mV]": [],
                "Sum_amplitude [mV]": [],
            },
        ),
    ],
)
def test_calculate_coincidence(
    A_time,
    A_ampl,
    B_time,
    B_ampl,
    C_time,
    C_ampl,
    D_time,
    D_ampl,
    coincidence_window,
    coincidence_citeria,
    expected,
):
    """
    Test the calculate_coincidence function.
    This function checks if the coincidence detection works correctly
    for various combinations of channels and criteria.

    Args:
        A_time: List of timestamps for channel A.
        A_ampl: List of amplitudes for channel A.
        B_time: List of timestamps for channel B.
        B_ampl: List of amplitudes for channel B.
        C_time: List of timestamps for channel C.
        C_ampl: List of amplitudes for channel C.
        D_time: List of timestamps for channel D.
        D_ampl: List of amplitudes for channel D.
        coincidence_window: Time window for coincidence detection.
        coincidence_citeria: List of criteria for coincidence and anti-coincidence.
        expected: Expected output as a dictionary.
    """
    result_df = prt.calculate_coincidence(
        A_time,
        A_ampl,
        B_time,
        B_ampl,
        C_time,
        C_ampl,
        D_time,
        D_ampl,
        coincidence_window,
        coincidence_citeria,
    )

    # Convert result DataFrame to dictionary for comparison
    result = {col: result_df[col].tolist() for col in result_df.columns}

    assert result == expected
