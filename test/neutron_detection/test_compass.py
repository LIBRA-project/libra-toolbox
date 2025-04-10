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
    

def check_dictionaries(test_dict, expected_dict):
    """ Tests single-layer dictionary (no dictionaries inside the dictionary)
    to make sure the keys, length, and overall numpy data is the same."""

    for key in expected_dict:
        assert key in test_dict
        assert len(test_dict[key]) == len(expected_dict[key])
        for a, b in zip(test_dict[key], expected_dict[key]):
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                assert np.array_equal(a, b)
            else:
                assert a == b


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
    check_dictionaries(data_filenames, expected_filenames)


@pytest.mark.parametrize("directory, expected_times, expected_energies",[
    (
        COMPASS_DATA_DIR, 
        {0: np.array([2382684499, 41455462001, 45918412749, 79592816499, 
                     112187341248, 115252007249, 128221445748, 143572718001, 
                     218888646140498, 218890578383748, 218901809939499, 218914735851248, 
                     218937251502999, 219016686095123]),
            1: np.array([18091371, 25234998, 52088998, 57670496, 
                        75054496, 82707949617246, 82707959048246, 82707982152002, 
                        82708016875123, 82708053920874, 82708055242374, 82708082392372,
                        82708137599746, 82708138860000, 162437014573246, 162437020032000,
                        162437036026247, 162437085015247, 162437092230746, 162437097488246,
                        162437145756003, 162437267549246, 162437270107246, 162437286368621,
                        162437307861498, 162437311924372])},
        {0:np.array([869, 611, 843, 372,
                     653, 582, 915, 202,
                     1135, 787, 1098, 808, 
                     702, 970]),
            1: np.array([811, 816, 288, 533,
                         362, 758, 806, 816,
                         821, 796, 692, 858, 
                         328, 809, 755, 438,
                         800, 580, 506, 522, 
                         370, 503, 799, 768,
                         407, 810])}
        )
])
def test_get_events(directory, expected_times, expected_energies):
    times, energies = compass.get_events(directory)
    assert isinstance(times, dict)
    assert isinstance(energies, dict)

    # Check if dictionaries have the same keys, length of data array, and 
    # the same overall data array

    check_dictionaries(times, expected_times)
    check_dictionaries(energies, expected_energies)


    
   
