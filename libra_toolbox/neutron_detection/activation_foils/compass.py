import numpy as np
import os


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
    return int(filename.split('@')[0][7:])


def sort_compass_files(directory: str) -> dict:
    """ Gets Compass csv data filenames
    and sorts them according to channel and ending number.
    The filenames need to be sorted by ending number because only
    the first csv file for each channel contains a header.
    
    Example of sorted filenames in directory: 
        1st file: Data_CH4@...22.CSV
        2nd file: Data_CH4@...22_1.CSV
        3rd file: Data_CH4@...22_2.CSV """

    filenames = os.listdir(directory)
    data_filenames = {}
    for filename in filenames:
        if filename.lower().endswith('.csv'):
            ch = get_channel(filename)
            # initialize filenames for each channel
            if ch not in data_filenames.keys():
                data_filenames[ch] = []

            data_filenames[ch].append(filename)
    # Sort filenames by number at end
    for ch in data_filenames.keys():
        data_filenames[ch] = np.sort(data_filenames[ch])

    return data_filenames




