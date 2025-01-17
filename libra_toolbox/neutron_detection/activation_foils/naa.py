import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import pickle
import datetime
import uproot
import glob

def get_decay_lines(nuclides):
    all_decay_lines = {'Ba133':{'energy':[80.9979, 276.3989, 302.8508, 356.0129, 383.8485],
                         'intensity':[0.329, 0.0716, 0.1834, 0.6205, 0.0894],
                         'half_life':[10.551*365.25*24*3600],
                         'activity_date':datetime.date(2014, 3, 19),
                         'activity':1 * 3.7e4},
                'Co60':{'energy':[1173.228, 1332.492],
                        'intensity':[0.9985, 0.999826],
                        'half_life':[1925.28*24*3600],
                        'actvity_date':datetime.date(2014, 3, 19),
                        'activity':0.872 * 3.7e4},
                'Na22':{'energy':[511, 1274.537],
                        'intensity':[1.80, 0.9994],
                        'half_life':[2.6018*365.25*24*3600],
                        'actvity_date':datetime.date(2014, 3, 19),
                        'activity': 5 * 3.7e4},
                'Cs137':{'energy':[661.657],
                         'intensity':[0.851],
                         'half_life':[30.08*365.25*24*3600],
                         'actvity_date':datetime.date(2014, 3, 19),
                         'activity':4.66 * 3.7e4},
                'Mn54':{'energy':[834.848],
                        'intensity':[0.99976],
                        'half_life':[312.20*24*3600],
                        'actvity_date':datetime.date(2016, 5, 2),
                        'activity':6.27 * 3.7e4}}
    decay_lines = {}
    for nuclide in nuclides:
        if nuclide in all_decay_lines.keys():
            decay_lines[nuclide] = all_decay_lines[nuclide]
        else:
            raise Warning('{} not yet added to get_decay_lines()'.format(nuclide))
    return decay_lines


def get_peak_inputs(samples):
    default_inputs = {'Na22':{'prom_factor':0.075, 'width':[10, 150], 'start_index':100},
               'Co60':{'prom_factor':0.2, 'width':[10, 150], 'start_index':400},
               'Ba133':{'prom_factor':0.1, 'width':[10, 200], 'start_index':100},
               'Mn54':{'prom_factor':0.2, 'width':[10, 100], 'start_index':100}}
    
    defaults = {'prom_factor':0.075, 'width':[10, 150], 'start_index':100}

    peak_inputs = {}
    for sample in samples:
        if sample in default_inputs.keys():
            peak_inputs[sample] = default_inputs[sample]
        else:
            peak_inputs[sample] = defaults
    return peak_inputs


def get_events(directory, skiprows=1):
    time_values = {}
    energy_values = {}
    for filename in os.listdir(directory):
        if filename.endswith('.CSV'):
            # Load data from the CSV file
            csv_file_path = os.path.join(directory, filename)

            # Determine digitizer channel
            ch = int(filename.split('@')[0][7:])
            if ch not in time_values.keys():
                time_values[ch] = []
                energy_values[ch] = []

            try:
                # data = np.genfromtxt(csv_file_path, delimiter=';', skip_header=1, missing_values=0)
                df = pd.read_csv(csv_file_path, delimiter=';', header=None, skiprows=skiprows)
                # data = df.to_numpy()
            except:
                raise Exception(f'Could not read in file: {csv_file_path}')

            # Assuming the energy values are in a specific column (e.g., column 1)
            time_column = 2
            energy_column = 3

            time_data = df[time_column].to_numpy()
            # print(time_data.shape)
            energy_data = df[energy_column].to_numpy()
            
            # Extract and append the energy data to the list
            time_values[ch].extend(time_data)
            # print(len(time_values[source]))
            energy_values[ch].extend(energy_data)
    for ch in time_values.keys():
        # sort data based on timestamp
        inds = np.argsort(time_values[ch])
        # convert times from picoseconds to seconds
        time_values[ch] = np.array(time_values[ch])[inds] /1e12
        energy_values[ch] = np.array(energy_values[ch])[inds]

    return energy_values, time_values


def get_start_stop_time(directory):
    info_file = os.path.join(directory, '../run.info')
    if os.path.isfile(info_file):
        time_format = "%Y/%m/%d %H:%M:%S.%f%z"
        with open(info_file, 'r') as file:
            continue_search = True
            while continue_search:
                line = file.readline()
                if 'time.start=' in line:
                    # get time string while cutting off '\n' newline
                    time_string = line.split('=')[1][:-1]
                    start_time = datetime.datetime.strptime(time_string, time_format)
                elif 'time.stop=' in line:
                    time_string = line.split('=')[1][:-1]
                    stop_time = datetime.datetime.strptime(time_string, time_format)
                    continue_search = False
                elif len(line) == 0:
                    continue_search = False
    else:
        raise LookupError('Could not find run.info file')
    
    return start_time, stop_time



def get_compass_counts(directories, bins=None, count_rate=False, skiprows=1, savefile=None):

    """ Obtain detector counts from .CSV file saved in CoMPASS."""

    get_raw_data = False
    
    if savefile is not None:
        if os.path.isfile(savefile):
            with open(savefile, 'rb') as file:
                counts = pickle.load(file)
        else:
            get_raw_data = True

    if get_raw_data:

        # Initialize an empty list to store energy values
        energy_values = {}
        time_values = {}

        counts = {}

        # Iterate through all CSV files in the directory
        for source in directories.keys():
            print('Reading in files for {}'.format(source))
            energy_values[source] = {}
            time_values[source] = {}
            counts[source] = {}
            for filename in os.listdir(directories[source]):
                if filename.endswith('.CSV'):
                    # Load data from the CSV file
                    csv_file_path = os.path.join(directories[source], filename)

                    # Determine digitizer channel
                    ch = int(filename.split('@')[0][7:])
                    if ch not in time_values[source].keys():
                        time_values[source][ch] = []
                        energy_values[source][ch] = []
                        counts[source][ch] = {}

                    try:
                        # data = np.genfromtxt(csv_file_path, delimiter=';', skip_header=1, missing_values=0)
                        df = pd.read_csv(csv_file_path, delimiter=';', header=None, skiprows=skiprows)
                        # data = df.to_numpy()
                    except:
                        raise Exception(f'Could not read in file: {csv_file_path}')

                    # Assuming the energy values are in a specific column (e.g., column 1)
                    time_column = 2
                    energy_column = 3

                    time_data = df[time_column].to_numpy()
                    # print(time_data.shape)
                    energy_data = df[energy_column].to_numpy()
                    erg_max = np.nanmax(energy_data)
                    if erg_max > 1e5:
                        print('time data may be in energy_data column for: \n', filename)
                        time_column = 4
                        energy_column = 5
                        time_data = df[time_column].to_numpy()
                        energy_data = df[energy_column].to_numpy()
                        erg_max = np.nanmax(energy_data)
                    
                    # Extract and append the energy data to the list
                    time_values[source][ch].extend(time_data)
                    # print(len(time_values[source]))
                    energy_values[source][ch].extend(energy_data)
                
                # Get real and live count times from ROOT file
                # (Note: the spectrum can also be obtained from the ROOT file)
                # elif filename.lower().endswith('.root'):
                #     root_file = uproot.open(os.path.join(directories[source], filename))
                #     for key in root_file.keys():
                #         if 'LiveTime' in key:
                #             split_key = key.split(';')[0]
                #             channel = int(split_key.split('_')[-1])
                #             counts[source][channel]['live_count_time'] = (
                #                 root_file[key].members['fMilliSec'] 
                #                 / 1000)
                #         if 'RealTime' in key:
                #             split_key = key.split(';')[0]
                #             channel = int(split_key.split('_')[-1])
                #             counts[source][channel]['real_count_time'] = (
                #                 root_file[key].members['fMilliSec']
                #                 / 1000)
            for ch in counts[source].keys():
                root_filename = glob.glob(os.path.join(directories[source], '*.root'))[0]
                root_file = uproot.open(root_filename)
                counts[source][ch]['live_count_time'] = (
                            root_file[f'LiveTime_{ch}'].members['fMilliSec'] 
                            / 1000)
                counts[source][ch]['real_count_time'] = (
                            root_file[f'RealTime_{ch}'].members['fMilliSec'] 
                            / 1000)
                

            # Create a histogram to represent the combined energy spectrum
            for ch in time_values[source].keys():
                # sort data based on timestamp
                inds = np.argsort(time_values[source][ch])
                # convert times to seconds
                time_values[source][ch] = np.array(time_values[source][ch])[inds] /1e12
                energy_values[source][ch] = np.array(energy_values[source][ch])[inds]
                # print(np.nanmax(energy_values[source]))

                energy_values[source][ch] = np.nan_to_num(energy_values[source][ch], nan=0)

                if isinstance(bins, int):
                    hist, bin_edges = np.histogram(energy_values[source][ch], bins=bins)
                elif bins=='double':
                    hist, bin_edges = np.histogram(energy_values[source][ch], bins=int(np.nanmax(energy_values[source][ch])/2))
                else:
                    # b = np.arange(0, max_channel[ch])
                    b = np.arange(0, np.max(energy_values[source][ch]))
                    hist, bin_edges = np.histogram(energy_values[source][ch], bins=b)

                total_time = np.max(time_values[source][ch]) - np.min(time_values[source][ch])
                # counts[source][ch]['count_time'] = total_time
                if np.abs(total_time - counts[source][ch]['real_count_time']) > 0.1:
                    print(f'Total time = {total_time}\n', 
                          'Real Count Time (root) = {}'.format(counts[source][ch]['real_count_time']))
                    raise Exception(f'Total time different from root file real count time')
                if count_rate:
                    counts[source][ch]['hist'] = hist / total_time
                else:
                    counts[source][ch]['hist'] = hist
                counts[source][ch]['bin_edges'] = bin_edges
            # Get count start time
            start_time, stop_time = get_start_stop_time(directories[source])
            for ch in counts[source].keys():
                counts[source][ch]['start_time'] = start_time
                counts[source][ch]['stop_time'] = stop_time
        
        
        # save data for faster opening in future
        if savefile is not None:
            with open(savefile, 'wb') as file:
                pickle.dump(counts, file)   

    return counts


def subtract_background(counts, background_directory, savefile=None):
    # Check if background subtracted counts have already been saved
    if savefile:
        if os.path.isfile(savefile):
            with open(savefile, 'rb') as file:
                counts = pickle.load(file)
            return counts
        
    energies, times = get_events(background_directory)
    b_count_time = {}
    for ch in times.keys():
        b_count_time[ch] = times[ch][-1] - times[ch][0]

    for sample in counts.keys():
        for ch in counts[sample].keys():
            if counts[sample][ch]['real_count_time'] < b_count_time[ch]:
                # get background counts for the duration of the sample count
                end_ind = np.nanargmin(np.abs(counts[sample][ch]['real_count_time'] - (times[ch] - times[ch][0])))
                b_hist, b_edges = np.histogram(energies[ch][:end_ind+1], bins=counts[sample][ch]['bin_edges'])
            else:
                b_hist, b_edges = np.histogram(energies[ch], bins=counts[sample][ch]['bin_edges'])
                b_hist = (b_hist * (counts[sample][ch]['real_count_time'] / b_count_time[ch]))

            counts[sample][ch]['hist'] = (counts[sample][ch]['hist'] - b_hist)
    if savefile:
        with open(savefile, 'wb') as file:
            pickle.dump(counts, file)


    return counts


def get_peaks(hist, source):
    start_index = 100
    prominence = 0.10 * np.max(hist[start_index:])
    height = 0.10 * np.max(hist[start_index:])
    width = [10, 150]
    indices = None
    distance = 30
    if 'na22' in source.lower():
        # find 511 keV peak first
        prominence = 0.01 * np.max(hist[start_index:])
        height = 0.9 * np.max(hist[start_index:])
        width = [10, 200]
    elif 'co60' in source.lower():
        start_index = 400 
        height = 0.60 * np.max(hist[start_index:])
        prominence = None
    elif 'ba133' in source.lower():
        width = [10, 200]
    elif 'mn54' in source.lower():
        height = 0.6 * np.max(hist[start_index:])
    peaks, peak_data = find_peaks(hist[start_index:], 
                                  prominence=prominence,
                                  height=height,
                                  width=width,
                                  distance=distance)
    peaks = np.array(peaks) + start_index
    if 'na22' in source.lower():
        # Find 1275 keV peak
        peak_511 = peaks[0]
        print(peak_511)
        start_index = peak_511 + 100
        print(start_index)
        prominence = 0.5 * np.max(hist[start_index:])
        height = 0.10 * np.max(hist[start_index:])

        high_peaks, peak_data = find_peaks(hist[start_index:], 
                                  prominence=prominence,
                                  height=height,
                                  width=width,
                                  distance=distance)
        high_peaks = np.array(high_peaks) + start_index
        print('Na22: searched prominence:', prominence, ' :', high_peaks, peak_data)
        peaks = [peak_511, high_peaks[0]]
    print(source, peaks)
    if indices:
        peaks = peaks[[indices]][0]

    # print(source, peaks)

    return peaks





def calibrate_counts(counts, decay_lines, peak_inputs=None, plot_calibration=False):

    calibration_energies = {}
    calibration_channels = {}
    coeff = {}

    if peak_inputs is None:
        peak_inputs = get_peak_inputs(decay_lines.keys())

    print(peak_inputs)

    # find what digitizer channels were used (ex. Ch0 and Ch1)
    ch_keys = counts[list(counts.keys())[0]].keys()
    for ch in ch_keys:
        calibration_energies[ch] = []
        calibration_channels[ch] = []

        for sample in decay_lines.keys():
            # print('\n', sample)
            if 'channel' in decay_lines[sample].keys():

                # If the peak chanels are already included in decay_lines, use them.
                calibration_channels[ch] += decay_lines[sample]['channel']
                calibration_energies[ch] += decay_lines[sample]['energy']
            else:
                # Use SciPy find_peaks()
                # peaks, peak_data = find_peaks(counts[sample][ch]['hist'][peak_inputs[sample]['start_index']:], 
                #                     prominence=peak_inputs[sample]['prom_factor']*np.max(counts[sample][ch]['hist'][peak_inputs[sample]['start_index']:]),
                #                     width=peak_inputs[sample]['width'])
                # peaks = np.array(peaks) + peak_inputs[sample]['start_index']
                peaks = get_peaks(counts[sample][ch]['hist'], sample)
                print(ch, sample, peaks)
                if len(peaks) != len(decay_lines[sample]['energy']):
                    raise LookupError('SciPy find_peaks() found {} photon peaks, while {} were expected'.format(len(peaks), 
                                                                                                                len(decay_lines[sample]['energy'])))
                calibration_channels[ch] += list(peaks)
                calibration_energies[ch] += decay_lines[sample]['energy']

                    
                    # print('Channel: ', calibration_channels[-1], ', Energy: ', calibration_energies[-1])
        inds = np.argsort(calibration_channels[ch])
        calibration_channels[ch] = np.array(calibration_channels[ch])[inds]
        calibration_energies[ch] = np.array(calibration_energies[ch])[inds]

        print(ch)
        print(calibration_channels[ch])
        print(calibration_energies[ch])

        # linear fit for calibration curve
        coeff[ch] = np.polyfit(calibration_channels[ch], calibration_energies[ch], 1)


        for source in counts.keys():
            counts[source][ch]['calibrated_bin_edges'] = np.polyval(coeff[ch], counts[source][ch]['bin_edges'])
        
        if plot_calibration:
            xs = np.linspace(np.min(calibration_channels[ch]), np.max(calibration_channels[ch]))
            ys = np.polyval(coeff[ch], xs)
            fig, ax = plt.subplots()
            ax.plot(calibration_channels[ch], calibration_energies[ch], '.', ms=10, label='Check Source Peaks')
            ax.plot(xs, ys, '-', label='Linear Fit')
            ax.set_xlabel('Channel')
            ax.set_ylabel('Energy [keV]')
            ax.set_title('Ch {}'.format(ch))
            ax.legend()
        
    return counts, coeff

def gauss1(x, H, m, A, x0, sigma): 
    return H + m * x + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss2(x, H, m, A1, x1, sigma1, A2, x2, sigma2):
    out = H + m * x + A1 * np.exp(-(x - x1) ** 2 / (2 * sigma1 ** 2)) \
        + A2 * np.exp(-(x - x2) ** 2 / (2 * sigma2 ** 2))
    return out

def gauss(x, b, m, *args):
    """ Creates a multipeak gaussian with a linear addition of the form:
    m * x + b + Sum_i (A_i * exp(-(x - x_i)**2) / (2 * sigma_i**2) """

    out = m * x + b
    if np.mod(len(args), 3) == 0:
        for i in range(int(len(args)/3)):
            out += args[i*3 + 0] * np.exp(-(x - args[i*3 + 1]) ** 2 / (2 * args[i*3 + 2] **2 ))
    else:
        raise ValueError('Incorrect number of gaussian arguments given.')
    return out


def get_singlepeak_area(hist, bins, peak_erg, search_width=300, plot=False):

    # get midpoints of every bin
    xvals = np.diff(bins)/2 + bins[:-1]

    peak_ind = np.argmin(np.abs((peak_erg) - xvals))
    search_start = np.argmin(np.abs((peak_erg - search_width/2) - xvals))
    search_end = np.argmin(np.abs((peak_erg + search_width/2) - xvals))

    slope_guess = (hist[search_end] - hist[search_start]) / (xvals[search_end] - xvals[search_start])

    guess_parameters = [0,
                        slope_guess,
                        hist[peak_ind],
                        peak_erg,
                        search_width/6]
    print(guess_parameters)
                        
    parameters, covariance = curve_fit(gauss1, xvals[search_start:search_end], hist[search_start:search_end], p0=guess_parameters) 
    print(parameters)

    mean = parameters[3]
    sigma = parameters[4]

    peak_start = np.argmin(np.abs((mean - 3*sigma) - xvals))
    peak_end = np.argmin(np.abs((mean + 3*sigma) - xvals))

    gross_area = np.trapz(hist[peak_start:peak_end], x=xvals[peak_start:peak_end])
    # trap_cutoff_area = (hist[peak_start] + hist[peak_end])/2 * (bins[peak_end] - bins[peak_start])
    trap_cutoff_area = np.trapz(parameters[0] + parameters[1] * xvals[peak_start:peak_end], x=xvals[peak_start:peak_end])
    area = gross_area - trap_cutoff_area

    if plot:
        fig, ax = plt.subplots()
        ax.stairs(hist, bins)
        ax.plot(xvals[search_start:search_end], gauss1(xvals[search_start:search_end], *parameters))
        # plot dashed vertical lines at edges of peak
        ax.plot([xvals[peak_start]]*2, [0, np.max(hist)], '--k')
        ax.plot([xvals[peak_end]]*2, [0, np.max(hist)], '--k')

        # trap_cutoff_slope = (hist[peak_end] - hist[peak_start]) / (bins[peak_end] - bins[peak_start])
        # trap_cutoff_ys = trap_cutoff_slope * (xvals[peak_start:peak_end] - bins[peak_start]) + hist[peak_start]
        trap_cutoff_ys = parameters[0] + parameters[1] * xvals[peak_start:peak_end]
        ax.fill_between(xvals[peak_start:peak_end], hist[peak_start:peak_end], trap_cutoff_ys, alpha=0.5)

        ax.set_xlabel('Energy [keV]')
        ax.set_title('Peak: {} keV'.format(peak_erg))
    
    
    return area


def get_multipeak_area(hist, bins, peak_ergs, search_width=600, plot=False):
    # get midpoints of every bin
    xvals = np.diff(bins)/2 + bins[:-1]

    search_start = np.argmin(np.abs((peak_ergs[0] - search_width/(2*len(peak_ergs))) - xvals))
    search_end = np.argmin(np.abs((peak_ergs[-1] + search_width/(2*len(peak_ergs))) - xvals))

    guess_slope = (hist[search_end] - hist[search_start]) / (xvals[search_end] - xvals[search_start])

    guess_parameters = [0, guess_slope]

    peak_inds = []
    for i in range(len(peak_ergs)):
        peak_ind =  np.argmin(np.abs((peak_ergs[i]) - xvals))
        guess_parameters += [hist[peak_ind],
                             peak_ergs[i],
                             search_width/(3 * len(peak_ergs))]

    # print(guess_parameters)
                        
    parameters, covariance = curve_fit(gauss, 
                                       xvals[search_start:search_end], 
                                       hist[search_start:search_end], 
                                       p0=guess_parameters) 
    # print(parameters)

    areas = []
    peak_starts = []
    peak_ends = []
    all_peak_params = []
    peak_amplitudes = []
    for i in range(len(peak_ergs)):
        peak_amplitudes += [parameters[2 + 3*i]]
        mean = parameters[2 + 3*i + 1]
        sigma = np.abs(parameters[2 + 3*i + 2])
        peak_start = np.argmin(np.abs((mean - 3*sigma) - xvals))
        peak_end = np.argmin(np.abs((mean + 3*sigma) - xvals))

        peak_starts += [peak_start]
        peak_ends += [peak_end]

        # Use unimodal gaussian to estimate counts from just one peak
        peak_params = [parameters[0], parameters[1], parameters[2 + 3*i], mean, sigma]
        all_peak_params += [peak_params]
        gross_area = np.trapz(gauss(xvals[peak_start:peak_end], *peak_params), 
                              x=xvals[peak_start:peak_end])

        # Cut off trapezoidal area due to compton scattering and noise
        trap_cutoff_area = np.trapz(parameters[0] + parameters[1] * xvals[peak_start:peak_end],
                                    x=xvals[peak_start:peak_end])
        area = gross_area - trap_cutoff_area
        areas += [area]

    if plot:
        colors = ['red', 'green', 'magenta', 'brown', 'cyan']
        fig, ax = plt.subplots(figsize=[5,4])
        ax.stairs(hist, bins)
        ax.plot(xvals, gauss(xvals, *parameters))

        erg_str = ''

        for i in range(len(peak_ergs)):
            # plot dashed vertical lines at edges of peak
            # ax.plot([xvals[peak_starts[i]]]*2, [0, np.max(hist)], '--', color=colors[i])
            # ax.plot([xvals[peak_ends[i]]]*2, [0, np.max(hist)], '--', color=colors[i])

            trap_cutoff_ys = parameters[0] + parameters[1] * xvals[peak_starts[i]:peak_ends[i]]
            print('Plot gauss params: {}'.format(all_peak_params[i]))
            ax.fill_between(xvals[peak_starts[i]:peak_ends[i]], gauss(xvals[peak_starts[i]:peak_ends[i]], *all_peak_params[i]), 
                        trap_cutoff_ys, color=colors[i], alpha=0.5)
            erg_str += ' {:.0f},'.format(peak_ergs[i])

        ax.set_ylabel('Counts', fontsize=14)
        ax.set_ylim(-np.max(peak_amplitudes)*0.2, np.max(peak_amplitudes)*1.5)
        ax.set_xlabel('Energy [keV]', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        ax.set_title('Peak:{} keV'.format(erg_str), fontsize=14)
    
    return areas



def group_close_values(data, threshold=200):
    # Sort the data to group values sequentially
    data.sort()
    
    # Initialize groups and a temporary group
    groups = []
    temp_group = [data[0]]
    
    for i in range(1, len(data)):
        # Check if the current value is within the threshold of the last value in the temp group
        if abs(data[i] - temp_group[-1]) < threshold:
            temp_group.append(data[i])
        else:
            # Commit the temp group to groups and start a new group
            groups.append(tuple(temp_group))
            temp_group = [data[i]]
    
    # Add the last group
    groups.append(tuple(temp_group))
    
    return groups


def get_peak_areas(hist, bins, peak_ergs, 
                   overlap_width=200, search_width=400, plot=False):

    areas = []

    # organize peak energies into tuples, in which peak energies close enough
    # to have overlapping peaks will be paired together
    erg_groups = group_close_values(peak_ergs, threshold=overlap_width)
    print(erg_groups)

    for erg_group in erg_groups:
            areas += get_multipeak_area(hist, bins, erg_group, 
                                        search_width=len(erg_group)*search_width,
                                        plot=plot)
            print(areas)
    return areas


def energy_efficiency(counts, decay_lines, nuclides=None,
                      plot_eff=True, plot_areas=False,
                      overlap_width=200, search_width=400,
                      degree=2, count_sum_peak=False,
                      ax_eff=None):
    if nuclides is None:
        nuclides = decay_lines.keys()

    effs = {}
    eff_errs = {}
    energies = {}
    for nuc in nuclides:
        sum_peak = False
        if len(decay_lines[nuc]['energy']) > 1 and count_sum_peak:
            peak_energies = decay_lines[nuc]['energy'] + [np.sum(decay_lines[nuc]['energy'])]
            sum_peak = True
        else:
            peak_energies = decay_lines[nuc]['energy']
        for ch in counts[nuc].keys():
            # initialize efficiency list
            if ch not in effs.keys():
                effs[ch] = []
                eff_errs[ch] = []
                energies[ch] = []

            print(nuc, ' Ch ', ch )
            areas = get_peak_areas(counts[nuc][ch]['hist'],
                            counts[nuc][ch]['calibrated_bin_edges'],
                            peak_energies,
                            overlap_width=overlap_width,
                            search_width=search_width,
                            plot=plot_areas)
            if sum_peak:
                areas = np.array(areas)[:-1] + areas[-1] / len(areas[:-1])
            print('Peak areas: ', areas)
            # measured activity
            # I think this should be divided by live count time, but maybe it should
            # be divided by real count time?? Should go over this again
            act_meas = np.array(areas) / (np.array(decay_lines[nuc]['intensity']) \
                                             * counts[nuc][ch]['live_count_time'] )
            print('Activity measured: ', act_meas)
            act_meas_err = np.sqrt(np.array(areas)) / (np.array(decay_lines[nuc]['intensity'])
                                             * counts[nuc][ch]['live_count_time'])
            # expected activity
            l = np.log(2) / decay_lines[nuc]['half_life']
            print('decay constant: ', l)
            time = (counts[nuc][ch]['start_time'] - decay_lines[nuc]['activity_date']).total_seconds()
            print('count time: ', time)
            act_expec = decay_lines[nuc]['activity'] * np.exp(-l * time)
            print('Activity expected: ', act_expec)
            print('efficiency: ', act_meas/act_expec)

            effs[ch] += list(act_meas / act_expec)
            eff_errs[ch] += list(act_meas_err / act_expec)
            energies[ch] += decay_lines[nuc]['energy']
    
    # Sort the data
    print('effs: ', effs)
    print('energies: ', energies)
    coeff = {}
    for ch in effs.keys():
        ind = np.argsort(energies[ch])
        energies[ch] = np.array(energies[ch])[ind]
        effs[ch] = np.array(effs[ch])[ind]
        eff_errs[ch] = np.array(eff_errs[ch])[ind]

        # Create polynomial fit
        coeff[ch] = np.polyfit(energies[ch], effs[ch], degree)
    
    if plot_eff:
        if ax_eff is None:
            fig, ax_eff = plt.subplots(nrows=1, ncols=len(effs.keys()), figsize=[10, 6])
        for i,ch in enumerate(effs.keys()):
            xvals = np.linspace(energies[ch][0], energies[ch][-1])
            ax_eff[i].errorbar(energies[ch], effs[ch]*100, eff_errs[ch]*100, fmt=".",
                         capsize=3)
            yvals = np.polyval(coeff[ch], xvals)
            ax_eff[i].plot(xvals, yvals*100, '--k')
            ax_eff[i].set_title('Ch {}'.format(ch))
            ax_eff[i].set_xlabel('Energy [keV]')
            ax_eff[i].set_ylabel('Detector Efficiency [%]')
    
    return effs, eff_errs, coeff
        






            


            
            
            
        
    



    



