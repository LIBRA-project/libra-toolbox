import pytest
import numpy as np
import os
from libra_toolbox.neutron_detection.activation_foils import compass

TEST_DIR = os.path.dirname(__file__)  # directory of this test file
COMPASS_DATA_DIR = os.path.join(TEST_DIR, "compass_data")


@pytest.mark.parametrize("filename, expected_channel", [
    ("Data_CH14@V1725_292_Background_250322.CSV", 14),
    ("Data_CH7@V1725_123_Background_250322.CSV", 7),
    ("Data_CH21@V1725_456_Background_250322.CSV", 21),
])


def test_get_channel(filename, expected_channel):
    ch = compass.get_channel(filename)
    assert ch == expected_channel

@pytest.mark.parametrize("directory, expected_filenames",[
    (COMPASS_DATA_DIR, {0:["Data_CH0@DT5730SB_Cs137_Problem_Waveform_1103.CSV",
                         "Data_CH0@DT5730SB_Cs137_Problem_Waveform_1103_1.CSV"],
                      1:["Data_CH1@V1725_Cs137_Normal_List_241107.CSV",
                         "Data_CH1@V1725_Cs137_Normal_List_241107_1.CSV",
                         "Data_CH1@V1725_Cs137_Normal_List_241107_2.CSV"]})
])
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


    
   
