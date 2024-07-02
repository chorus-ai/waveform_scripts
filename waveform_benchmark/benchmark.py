#!/usr/bin/python3

import importlib
import os
import random
import sys
import tempfile

import numpy as np

from waveform_benchmark.input import load_wfdb_signals
from waveform_benchmark.ioperf import PerformanceCounter
from waveform_benchmark.utils import repeat_test
from waveform_benchmark.utils import median_attr

from memory_profiler import memory_usage
import time


def append_result(format_name, waveform_name, test_name, result, format_list, waveform_list, test_list, result_list):
    """
    Add a result to the summary lists
    """
    format_list.append(format_name)
    waveform_list.append(waveform_name)
    test_list.append(test_name)
    result_list.append(result)

    return format_list, waveform_list, test_list, result_list

def _run_read_test(fmt, path, total_length, all_channels, block_length, block_count,
                   test_min_dur = 10, test_min_iter = 3):
    counters = []
    for i in repeat_test(test_min_dur, test_min_iter):
        r = random.Random(12345)
        with PerformanceCounter() as pc:
            for j in range(block_count):
                t0 = r.random() * (total_length - block_length)
                t1 = t0 + block_length
                fmt().read_waveforms(path, t0, t1, all_channels) 
        counters.append(pc)
    return counters
    


def _run_read_test_1channel(fmt, path, total_length, all_channels, block_length, block_count,
                            test_min_dur = 10, test_min_iter = 3):
    counters = []
    for i in repeat_test(test_min_dur, test_min_iter):
        r = random.Random(12345)
        with PerformanceCounter() as pc:
            for j in range(block_count):
                t0 = r.random() * (total_length - block_length)
                t1 = t0 + block_length
                fmt().read_waveforms(path, t0, t1, all_channels)    
        counters.append(pc)
    return counters

def compute_snr(reference_signal, output_signal):
    """
    Compute the signal-to-noise ratio (SNR) for the signal in decibels.
    """

    # Convert the data to NumPy arrays as needed.
    reference_signal = np.asarray(reference_signal)
    output_signal = np.asarray(output_signal)

    # Check that the signals have the same dimensions and all finite values.
    assert(np.array_equal(np.shape(reference_signal), np.shape(output_signal)))
    assert(np.all(np.isfinite(reference_signal)) and np.all(np.isfinite(output_signal)))

    # Compute the noise in the signal.
    noise_signal = output_signal - reference_signal

    # Compute the SNR with special handling for edge cases.
    x = np.sum(reference_signal**2)
    y = np.sum(noise_signal**2)

    if x > 0 and y > 0:
        snr = 10 * np.log10(x / y)
    elif y == 0:
        snr = float('inf')
    else:
        snr = float('nan')

    return snr


def run_benchmarks(input_record, format_class, pn_dir=None, format_list=None, waveform_list=None, test_list=None,
                   result_list=None):

    # Load the class we will be testing
    module_name, class_name = format_class.rsplit('.', 1)
    module = importlib.import_module(module_name)
    fmt = getattr(module, class_name)

    # Load the example data
    input_record = input_record.removesuffix('.hea')
    if pn_dir:
        waveforms = load_wfdb_signals(input_record, pn_dir)
    else:
        waveforms = load_wfdb_signals(input_record)
    all_channels = list(waveforms.keys())

    total_length = 0
    timepoints_per_second = 0
    actual_samples = 0
    waveform_characterizations = {}
    for i, (name, waveform) in enumerate(waveforms.items()):
        precise_channel_length = 0
        channel_length = waveform['chunks'][-1]['end_time']
        total_length = max(total_length, channel_length)
        timepoints_per_second += waveform['samples_per_second']
        actual_samples += sum(len(chunk['samples'])
                              for chunk in waveform['chunks'])

        # Collect summary information about each channel in the waveform
        inv_gain = 1/waveform['chunks'][-1]['gain']
        res_rounded = f'{float(f"{inv_gain:.3g}"):g}'
        resolution = f"{res_rounded}({waveforms[name]['units']})"
        precise_channel_length += sum(chunk['end_time'] - chunk['start_time']
                              for chunk in waveform['chunks'])
        waveform_characterizations[name] = {
        'fs': waveform['samples_per_second'],
        'bit_resolution': resolution,
        'channel_length': precise_channel_length
        }

    total_timepoints = total_length * timepoints_per_second
    if pn_dir:
        record_name = os.path.join(pn_dir, input_record)
    else:
        record_name = input_record

    TEST_BLOCK_LENGTHS = [
        [total_length, 1],
        [500, 5],               # 5 random blocks of 500 seconds
        [50, 50],               # 50 random blocks of 50 seconds
        [5, 500],               # 500 random blocks of 5 seconds
    ]

    TEST_MIN_DURATION = 10
    TEST_MIN_ITERATIONS = 3

    print('_' * 64)
    print('Format: %s' % format_class)
    if fmt.__doc__:
        print('         (%s)'
              % fmt.__doc__.strip().splitlines()[0].rstrip('.'))

    print('Record: %s' % record_name)
    print('         %.0f seconds x %d channels'
          % (total_length, len(all_channels)))
    print('         %d timepoints, %d samples (%.1f%%)'
          % (total_timepoints, actual_samples,
             100 * actual_samples / total_timepoints))
    print('_' * 64)

    # Print summary information for each waveform channel
    print('Channel summary information:')
    print(f" {'signal':<12} {'fs(Hz)':<10} {'Bit resolution':<20} {'Channel length(s)':<20}")
    for (signal, values) in waveform_characterizations.items():
        fs_value = f"{values['fs']:.2f}"
        bit_resolution_value = values['bit_resolution']
        channel_length_value = f"{values['channel_length']:.0f}"
        print(f" {signal:<12} {fs_value:<10} {bit_resolution_value:<20} {channel_length_value:<20}")
    print('_' * 64)

    with tempfile.TemporaryDirectory(prefix='wavetest-', dir='.') as tempdir:
        path = os.path.join(tempdir, 'wavetest')

        # Write the example data to a file or files.
        time1 = time.time()
        with PerformanceCounter() as pc_write:
            # if (MEM_METHOD == 0):
            mem_usage = memory_usage((fmt().write_waveforms, (path, waveforms), {}), include_children = True, max_usage = True)
            # else:
            #     fmt().write_waveforms(path, waveforms)
        wall_time = time.time() - time1
        
        # Calculate total size of the file(s).
        output_size = 0
        for subdir, dirs, files in os.walk(tempdir):
            for file in files:
                output_size += os.path.getsize(os.path.join(subdir, file))

        print('Output size:    %.0f KiB (%.2f bits/sample)'
              % (output_size / 1024, output_size * 8 / actual_samples))
        print('Time to output: %.0f sec' % pc_write.cpu_seconds)
        print('Wall Time: %.0f s' % wall_time)
                
        print('Memory Used (memory_profiler): %.0f MiB' % mem_usage)
        print('Maximum Memory Used (max_rss): %.0f MiB' % pc_write.max_rss)
        print('Memory Malloced (tracemalloc): %.0f MiB' % pc_write.malloced)
        
        
        print('_' * 64)

        if format_list is not None:
            # Append output size and write time
            format_list, waveform_list, test_list, result_list = append_result(format_class, input_record,
                                                                               'output_size',
                                                                               (output_size / 1024), format_list,
                                                                               waveform_list, test_list, result_list)
            format_list, waveform_list, test_list, result_list = append_result(format_class, input_record,
                                                                               'output_time',
                                                                               pc_write.cpu_seconds, format_list,
                                                                               waveform_list, test_list, result_list)

        # Fidelity Check
        # Loop over each waveform
        print("Fidelity check:")
        print()
        print("Chunk\t\t\tNumeric Samples\t\t\t\t  NaN Samples")
        print(f"\t# Errors  /  Total\t{'% Eq':^8}\t{'SNR':^8}\tNaN Values Match")

        for channel,waveform in waveforms.items():
            print(f"Signal: {channel}")
            # Loop over chunks
            # print("Chunk\t\t Numeric Samples\t\t  NaN Samples")
            # print(f"\t# Errors  /  Total\t{'% Eq':^8}\tNaN Values Match")

            for i_ch, chunk in enumerate(waveform["chunks"]):
                st = chunk["start_time"]
                et = chunk["end_time"]
                data = chunk["samples"]

                # read chunk from file
                filedata = fmt().read_waveforms(path, st, et, [channel])
                filedata = filedata[channel]
                
                # compare values

                # check arrays are same size
                if data.shape != filedata.shape:
                    print(f"{i_ch:^5}\t --- Different shapes (input: {data.shape}, file: {filedata.shape}) ---")
                    continue

                # check for nans in correct location
                NANdiff = np.sum(np.isnan(data) != np.isnan(filedata))
                numnan = np.sum(np.isnan(data))
                numnanstr = f"{'N' if NANdiff else 'Y'} ({numnan})"

                # remove nans for equality check
                data_nonan = data[~np.isnan(data)]
                filedata_nonan = filedata[~np.isnan(data)]

                # use numpy's isclose to determine floating point equality
                isgood = np.isclose(filedata_nonan, data_nonan, atol=0.5/chunk['gain'])
                numgood = np.sum(isgood)
                fpeq_rel = numgood/len(data_nonan)

                # compute SNR to quantify signal fidelity
                snr = compute_snr(data_nonan, filedata_nonan)

                # print to table
                print(f"{i_ch:^5}\t{len(data_nonan)-numgood:10}/{len(data_nonan):10}\t{fpeq_rel*100:^6.3f}\t\t{snr:^6.1f}\t\t{numnanstr:^16}")

                # print up to 10 bad values if not all equal
                if numgood != len(data_nonan):
                    print("Subset of unuequal numeric data from input:")
                    print(data_nonan[~isgood][:10])
                    print("Subset of unuequal numeric data from formatted file:")
                    print(filedata_nonan[~isgood][:10])
                    print(f"(Gain: {chunk['gain']})")
            # print('_' * 64)
        print('_' * 64)
        print('Read performance (median of N trials):')
        print(' #seek  #read      KiB      sec   walltime      Mem MB (used/maxrss/malloced)    [N]')

        for block_length, block_count in TEST_BLOCK_LENGTHS:
            
            time1 = time.time()
            counters = []
            mem_usage, counters = memory_usage((_run_read_test, (fmt, path, total_length, all_channels, block_length, block_count), {'test_min_dur': TEST_MIN_DURATION, 'test_min_iter': TEST_MIN_ITERATIONS}), include_children = True, max_usage = True, retval = True)
            walltime = time.time() - time1
            
            print('%6.0f %6.0f %8.0f %8.4f   %8.4f   %8.4f/%8.4f/%8.4f %6s read %d x %.0fs, all channels'
                  % (median_attr(counters, 'n_seek_calls'),
                     median_attr(counters, 'n_read_calls'),
                     median_attr(counters, 'n_bytes_read') / 1024,
                     median_attr(counters, 'cpu_seconds'),
                     walltime / len(counters),
                     mem_usage,
                     median_attr(counters, 'max_rss'),
                     median_attr(counters, 'malloced'),
                     '[%d]' % len(counters),
                     block_count,
                     block_length))

            if format_list is not None:
                # Append read time result
                format_list, waveform_list, test_list, result_list = append_result(format_class, input_record,
                                                                                   f'{block_count}_all',
                                                                                   median_attr(counters, 'cpu_seconds'),
                                                                                   format_list,
                                                                                   waveform_list, test_list,
                                                                                   result_list)

        for block_length, block_count in TEST_BLOCK_LENGTHS:
            time1 = time.time()
            counters = []
            mem_usage, counters = memory_usage((_run_read_test_1channel, (fmt, path, total_length, all_channels, block_length, block_count), {'test_min_dur': TEST_MIN_DURATION, 'test_min_iter': TEST_MIN_ITERATIONS}), include_children = True, max_usage = True, retval = True)
            walltime = time.time() - time1
            
            print('%6.0f %6.0f %8.0f %8.4f   %8.4f   %8.4f/%8.4f/%8.4f %6s read %d x %.0fs, one channel'
                  % (median_attr(counters, 'n_seek_calls'),
                     median_attr(counters, 'n_read_calls'),
                     median_attr(counters, 'n_bytes_read') / 1024,
                     median_attr(counters, 'cpu_seconds'),
                     walltime / len(counters),
                     mem_usage,
                     median_attr(counters, 'max_rss'),
                     median_attr(counters, 'malloced'),
                     '[%d]' % len(counters),
                     block_count,
                     block_length))

            if format_list:
                format_list, waveform_list, test_list, result_list = append_result(format_class, input_record,
                                                                                   f'{block_count}_one',
                                                                                   median_attr(counters, 'cpu_seconds'),
                                                                                   format_list,
                                                                                   waveform_list, test_list,
                                                                                   result_list)

    print('_' * 64)

    if format_list is not None:
        # Return the lists with appended results for this waveform
        return format_list, waveform_list, test_list, result_list
