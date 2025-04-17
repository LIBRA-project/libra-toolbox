import pytest
import numpy as np
import os
from libra_toolbox.neutron_detection.activation_foils import compass

TEST_DIR = os.path.dirname(__file__)  # directory of this test file
COMPASS_DATA_DIR = os.path.join(TEST_DIR, "compass_test_data")
RUN_1_TEST_DATA_DIR = os.path.join(COMPASS_DATA_DIR, "baby_1L_run1_test_data")
RUN_3_TEST_DATA_DIR = os.path.join(COMPASS_DATA_DIR, "baby_1L_run3_test_data")


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


@pytest.mark.parametrize(
    "directory, expected_filenames",
    [
        (
            os.path.join(RUN_1_TEST_DATA_DIR, "Background_20241108_0027/UNFILTERED"),
            {2: ["Data_CH2@V1725_292_Background_20241108_0027.CSV"]},
        ),
        (
            os.path.join(
                RUN_1_TEST_DATA_DIR, "Cs137_4_66uCi_19Mar2014_20241107/UNFILTERED"
            ),
            {
                2: [
                    "Data_CH2@V1725_292_Cs137_4_66uCi_19Mar2014_241107.CSV",
                    "Data_CH2@V1725_292_Cs137_4_66uCi_19Mar2014_241107_1.CSV",
                    "Data_CH2@V1725_292_Cs137_4_66uCi_19Mar2014_241107_2.CSV",
                    "Data_CH2@V1725_292_Cs137_4_66uCi_19Mar2014_241107_3.CSV",
                    "Data_CH2@V1725_292_Cs137_4_66uCi_19Mar2014_241107_4.CSV",
                    "Data_CH2@V1725_292_Cs137_4_66uCi_19Mar2014_241107_5.CSV",
                    "Data_CH2@V1725_292_Cs137_4_66uCi_19Mar2014_241107_6.CSV",
                    "Data_CH2@V1725_292_Cs137_4_66uCi_19Mar2014_241107_7.CSV",
                ]
            },
        ),
        (
            os.path.join(
                RUN_3_TEST_DATA_DIR, "Na22_9_98uCi_29Sep2023_250318_run3/UNFILTERED"
            ),
            {
                4: [
                    "Data_CH4@V1725_292_Na22_9_98uCi_29Sep2023_250318_run3.CSV",
                    "Data_CH4@V1725_292_Na22_9_98uCi_29Sep2023_250318_run3_1.CSV",
                    "Data_CH4@V1725_292_Na22_9_98uCi_29Sep2023_250318_run3_2.CSV",
                ],
                5: [
                    "Data_CH5@V1725_292_Na22_9_98uCi_29Sep2023_250318_run3.CSV",
                    "Data_CH5@V1725_292_Na22_9_98uCi_29Sep2023_250318_run3_1.CSV",
                ],
            },
        ),
    ],
)
def test_sort_compass_files(directory, expected_filenames):
    data_filenames = compass.sort_compass_files(directory)
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
