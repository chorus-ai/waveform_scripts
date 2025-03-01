import argparse
import csv
import os
import sys

import pandas as pd
from tqdm import tqdm

from waveform_benchmark.benchmark import run_benchmarks


def read_csv(file_path):
    """
    Read in a csv file with a header
    """
    data = []
    with open(file_path, 'r', newline='') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            data.append(row)

    return data


def init_summary_file(summary_file):
    """
    Initialize the summary lists - either grab existing data from a file to be appended to or create empty lists for all
    new data
    """
    if os.path.exists(summary_file):
        df_summary = pd.read_csv(summary_file)
        try:
            format_list = list(df_summary['format'])
            waveform_list = list(df_summary['waveform'])
            test_list = list(df_summary['test'])
            result_list = list(df_summary['result'])
        except:
            print("Required columns not found in the waveform suite summary file")
            sys.exit(1)
    else:
        format_list = []
        waveform_list = []
        test_list = []
        result_list = []

    return format_list, waveform_list, test_list, result_list


def save_summary(format_list, waveform_list, test_list, result_list, summary_file):
    """
    Save a summary of the results to a CSV file
    """
    df_updated_summary = pd.DataFrame(zip(format_list, waveform_list, test_list, result_list),
                                      columns=['format', 'waveform', 'test', 'result'])

    # Add columns for the last identifier for format and waveform
    df_updated_summary['format_id'] = df_updated_summary['format'].str.split('.').str[-1]
    df_updated_summary['waveform_id'] = df_updated_summary['waveform'].str.split('/').str[-1]

    # Reorder the columns
    df_updated_summary = df_updated_summary[['test', 'waveform', 'waveform_id', 'format', 'format_id', 'result']]

    df_updated_summary.to_csv(summary_file, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_record', '-r',
                    help='The record name to run benchmarking against')
    ap.add_argument('--format_class', '-f',
                    help='The format to save the waveform in')
    ap.add_argument('--physionet_directory', '-p',
                    default=None,
                    help='The physionet database directory to read the source waveform from')
    ap.add_argument('--save_output_to_log', '-l',
                    action='store_true',
                    help='Save all of the benchmarking results to a log file')
    ap.add_argument('--waveform_suite_table', '-s',
                    default=None,
                    help='A csv table with input_record, physionet directory, and format class for multiple files')
    ap.add_argument('--waveform_suite_summary_file', '-w',
                    default='waveform_suite_benchmark_summary.csv',
                    help='Save a CSV summary of the waveform suite run to this path/file')
    ap.add_argument('--test_only', 
                    default=False, action='store_true',
                    help='Run only the tests, do not run the benchmarks')
    ap.add_argument('--memory_profiling', '-m',
                    default=False, type=bool, action=argparse.BooleanOptionalAction,
                    help='Run memory profiling on the benchmarking process')
    ap.add_argument('--verbose', '-v',
                    action='store_true',
                    help='Add additional information to output')
    opts = ap.parse_args()

    # If log is requested send the output there
    if opts.save_output_to_log:
        log_file = open('benchmark_results.log', 'a')

        # Send the output to the log file
        sys.stdout = log_file

    # Check conditions based on the parsed arguments
    if not opts.waveform_suite_table:
        # If a waveform suite table is not provided, input_record and format_class must be provided
        if opts.input_record is None or opts.format_class is None:
            ap.error('--input_record and --format_class are required unless --waveform_suite_table is specified')

    # If a table with multiple files is passed we loop through it and save a summary of the results
    if opts.waveform_suite_table:
        waveform_suite = read_csv(opts.waveform_suite_table)

        format_list, waveform_list, test_list, result_list = init_summary_file(opts.waveform_suite_summary_file)

        for waveform_file in tqdm(waveform_suite):
            # Extract metadata from looped file and launch benchmarking
            record = waveform_file[0]
            format = waveform_file[1]
            pn_dir = waveform_file[2]
            format_list, waveform_list, test_list, result_list = run_benchmarks(input_record=record,
                                                                                format_class=format, pn_dir=pn_dir,
                                                                                format_list=format_list,
                                                                                waveform_list=waveform_list,
                                                                                test_list=test_list,
                                                                                result_list=result_list,
                                                                                test_only = opts.test_only,
                                                                                mem_profile = opts.memory_profiling,
                                                                                verbose = opts.verbose)

        save_summary(format_list, waveform_list, test_list, result_list, opts.waveform_suite_summary_file)

    # Run benchmarking against a single file
    else:
        run_benchmarks(input_record=opts.input_record,
                       format_class=opts.format_class,
                       pn_dir=opts.physionet_directory,
                       test_only = opts.test_only,
                       mem_profile = opts.memory_profiling,
                       verbose = opts.verbose)

    # Close the log file after the run is complete
    if opts.save_output_to_log:
        log_file.close()


if __name__ == '__main__':
    main()
