"""
This module provides a wrapper class for applying the machine learning (ML) phase picking module Quakephase.

Authors:
- Noah Munro-Kagan (noahmunrokagan@gmail.com)

Created: June 26, 2024
Last Modified: July 3, 2024

Dependencies:
- numpy
- h5py
- obspy.core
- seisbench
- gc
- sys
- os
- psutil
- time
- matplotlib.pyplot
- scipy.signal
- concurrent.futures
- quakephase
"""
import concurrent.futures
import os
import numpy as np
import h5py
import time
import sys
import psutil
from obspy.core import Stream, Trace, UTCDateTime

from quakephase import quakephase

import seisbench
import gc

#plotting imports
import matplotlib.pyplot as plt

from obspy import read
from scipy.signal import find_peaks

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the system path
sys.path.append(parent_dir)

# Now you can import the quakephase module
from quakephaseNoah import quakephaseLBQ, quakephaseLBQParallel

class AcousticEmissionWrapper:
    """
    A wrapper class for managing acoustic emission (AE) data and implementing Quakephase.
    
    Attributes:
    - sensors (list): The names of each AE sensor whos data will be loaded.
    - samplingRate (int): The sampling rate of the AE sensors.
    - channel (str, optional): AE sensors' channel code.
    - data_dict (dict): Loaded raw data from AE sensors.

    Authours:
    - Noah Munro-Kagan (noahmunrokagan@gmail.com)

    Example:
        sensors = ['A1', 'A3', 'B1', 'B3', 'C1', 'C3', 'D1', 'D3', 'E1', 'E3', 'F1', 'F3', 'G1', 'G3', 'H1', 'H3']
        wrapper = AcousticEmissionWrapper(sensors = sensors, samplingRate = 10e6, channel = 'FPZ')

    """
    def __init__(self, sensors: list, samplingRate: int, channel: str = None):
        """
        Initialized the AcousticEmissionWrapper with a AE sensors, the sampling rate and a channel name
        
        Parameters:
        - sensors (list): The names of each AE sensor whos data will be loaded.
        - samplingRate (int): The sampling rate of the AE sensors.
        - channel (str, optional): AE sensors' channel code.
        
        Returns:
        None
        """
        self.sensors = sensors
        self.samplingRate = samplingRate
        self.channel = channel
        self.data_dict = {}
        self.global_starttime = UTCDateTime()

    def getSensors(self) -> list:
        return self.sensors

    def load_data(self, path: str, isDirectory: bool = False, num_batches: int = 1):
        """
        Loads data from input file or directory into an Obspy Stream with a single Obspy Trace for each AE sensor and\
            adds them to self.data_dict, can be batched. Files can be in .txt, .tpc5 or .mat format.
        
        Parameters:
        path (str): Absolute path to the data file or directory.
        isDirectory (bool): Indication if path leads to a directory of files.
        num_batches (int, optional): Indication of how many separate Streams to split the input data into.

        Returns:
        None

        Raises:
        - ValueError: If the parameters were incorrect.
        - SyntaxError: If an unknown error occured.

        Example:
        path = /path/to/data/file.txt
        sensors = ['A1', 'A3', 'B1', 'B3', 'C1', 'C3', 'D1', 'D3', 'E1', 'E3', 'F1', 'F3', 'G1', 'G3', 'H1', 'H3']
        wrapper = AcousticEmissionWrapper(sensors = sensors, samplingRate = 10e6, channel = 'FPZ')
        wrapper.load_data(path=path)

        Authours:
        - Noah Munro-Kagan (noahmunrokagan@gmail.com) 
        """

        if not path or not path.strip():
            raise ValueError("Input path cannot be empty")
        if isDirectory and os.path.isfile(path):
            raise ValueError("path cannot lead to a file if isDirectory == True")
        if num_batches < 1 or not isinstance(num_batches, int):
            raise ValueError("num_batches must be an integer larger than 0")
        if isDirectory and num_batches > 1:
            raise ValueError("Cannot batch a directory's data")
        

        if not isDirectory:
            self._load_data_file(path=path, num_batches=num_batches)
        elif isDirectory:
            self._load_data_directory(path)
        else:
            raise SyntaxError("Unknown error occurred, please check your input for load_data")


    def _load_data_file(self, path: str, num_batches: int):
        """
        Load data from a file. This method is intended for internal use only
        and should only be called by the 'load_data' or '_load_data_directory' methods.

        Parameters:
        - path (str): The path to the data file.

        Returns:
        None

        Raises:
        - TypeError: If the data file is not in the format .txt, .mat or .tpc5.
        """
        print('Loading data from a file')

        file_ending = path.split('.')[-1]

        if file_ending == 'txt':
            self._load_data_txt(path)
        elif file_ending == 'mat':
            self._load_data_mat(path)
        elif file_ending == 'tpc5':
            self._load_data_tpc5(path=path, num_batches=num_batches)
        else:
            raise TypeError("Input file must be .mat, .txt or .tpc5 file")

        
    def _load_data_txt(self, path):
        """
        Load data from a .txt file to a single Obspy Trace. This method is intended for internal use 
        only. Should only be called by the '_load_data_file' method.

        Parameters:
        - path (str): The path to a .txt file

        Returns:
        None
        """
        base_name = os.path.basename(path)
        base_name_split = base_name.split('_')
        eventNumber = int((base_name_split[-1].split('.')[0].lstrip('event')).lstrip('0'))
        sensorName = base_name_split[-2]

        traceData = np.genfromtxt(path, delimiter=',', skip_header=1)
        trace = Trace(data=traceData[:, 1])
        trace.stats.station = sensorName
        trace.stats.sampling_rate = self.samplingRate
        trace.stats.channel = self.channel

        print(f'Loaded trace for event {eventNumber} from sensor {sensorName}')

        #garbage collection, efficent memory usage
        del traceData
        self._clear_memory()

        return trace

    def _load_data_mat(self, path):
        print('got inside load_dataMat, implementation needed')
        # Implementation needed

    def _load_data_tpc5(self, path: str, num_batches: int):
        """
        Load data from a .tpc5 file. This method is intended for internal use only and should 
        only be called by the '_load_data_file' or '_load_data_directory' methods.

        Parameters:
        path (str): Path to the .tpc5 file
        num_batches (int): Number of batches to divide the data into. Defaults to None (no batching).

        Returns:
        None
        """
        print('Loading data from .tpc5 file')
        if num_batches == 1:
            self._load_data_tpc5_full(path, num_batches)
        else:
            self._load_data_tpc5_batched(path=path, num_batches=num_batches)

    def _load_data_tpc5_full(self, path: str, num_batches: int):
        """
        Load data from a .tpc5 file. This method is intended for internal use only and should 
        only be called by the '_load_data_tpc5' method.

        Parameters:
        path (str): Path to the .tpc5 file.
        num_batches (int): Number of Streams to split .tpc5 file into.

        Returns:
        None
        """
        with h5py.File(path, 'r') as hdf5File:
            channelPathTemplate = '/measurements/00000001/channels/{:08d}/blocks/00000001/data'

            # Determine the total number of samples (assume all sensors have the same number of samples)
            total_samples = hdf5File[channelPathTemplate.format(1)].shape[0]

        stream = self._load_data_tpc5_internal(path=path, num_samples_per_batch=total_samples, batch_index=0)
        base_name = os.path.basename(path)
        file_number = ((base_name.split('_')[-1]).split('.')[0]).lstrip('tpc5')
        self.data_dict['full'] = stream
        print(f'Loaded data for parsed .tpc5 file {file_number}')

        del stream
        self._clear_memory()

    def _load_data_tpc5_batched(self, path, num_batches):
        """
        Load and batch data from a .tpc5 file. This method is intended for internal use only and should 
        only be called by the '_load_data_tpc5' method.

        Parameters:
        path (str): Path to the .tpc5 file.
        num_batches (int): Number of streams to batch the data into.

        Returns:
        None
        """
        print(f'loading tpc5 file in {num_batches} batches')
        base_name = os.path.basename(path)
        file_number = ((base_name.split('_')[-1]).split('.')[0]).lstrip('tpc5')
        
        # Open .tpc5 file
        with h5py.File(path, 'r') as hdf5File:
            channelPathTemplate = '/measurements/00000001/channels/{:08d}/blocks/00000001/data'

            # Determine the total number of samples (assume all sensors have the same number of samples)
            total_samples = hdf5File[channelPathTemplate.format(1)].shape[0]
            num_samples_per_batch = total_samples // num_batches  # integer division

            for batch_index in range(num_batches):
                stream = self._load_data_tpc5_internal(path, num_samples_per_batch=num_samples_per_batch, hdf5File=hdf5File, batch_index=batch_index)
                self.data_dict[str(batch_index)] = stream
                print(f'Loaded batch {batch_index + 1}/{num_batches} for .tpc5 file {file_number}')

                # Clearing local memory of unneeded data
                del stream
                self._clear_memory()

    def _load_data_tpc5_internal(self, path, num_samples_per_batch: int, batch_index: int, hdf5File=None):
        """
        Load data from a .tpc5 file. This method is intended for internal use only and should 
        only be called by the '_load_data_tpc5_full' or '_load_data_tpc5_batched' methods.

        Parameters:
        path (str): Path to the .tpc5 file.
        num_samples_per_batch (int): Number of samples to include in the stream.
        batch_index (int): Batch that is being loaded into a Stream.
        hdf5File (h5py File object): Loaded HDF5 file

        Returns:
        - stream: An obspy stream containing AE data from the .tpc5 file

        Raises:
        - Exception: Unforseen exception.
        """
        try:
            print(f"Loading data batch {batch_index} from {path}...")
            if hdf5File is None:
                with h5py.File(path, 'r') as hdf5File:
                    stream = self._process_tpc5_file(hdf5File=hdf5File, num_samples_per_batch=num_samples_per_batch, batch_index=batch_index)
            else:
                stream = self._process_tpc5_file(hdf5File, num_samples_per_batch, batch_index)

            print(f"Data batch {batch_index} loaded successfully")
            return stream
        except Exception as e:
            print(f"Error loading data for batch {batch_index}: {e}")
            return None





    def _process_tpc5_file(self, hdf5File, num_samples_per_batch: int, batch_index: int):
        """
        Load data from a .tpc5 file. This method is intended for internal use only and should 
        only be called by the '_load_data_tpc5_internal' method.

        Parameters:
        num_samples_per_batch (int): Number of samples to include in the stream.
        batch_index (int): Batch that is being loaded into a Stream.
        hdf5File (h5py File object): Loaded HDF5 file.

        Returns:
        - stream: An obspy stream containing AE data from the .tpc5 file
        """
        
        stream = Stream()
        channelPathTemplate = '/measurements/00000001/channels/{:08d}/blocks/00000001/data'
        

        
        start_sample = np.int64(batch_index * num_samples_per_batch)
        end_sample = np.int64(min((batch_index + 1) * num_samples_per_batch, hdf5File[channelPathTemplate.format(1)].shape[0]))

        for sensorIndex in range(1, len(self.sensors) + 1):
            channelPath = channelPathTemplate.format(sensorIndex)
            channelPathTime = channelPath.strip('data')
            traceData = np.array(hdf5File[channelPath][start_sample:end_sample])

            startTime = hdf5File[channelPathTime].attrs['startTime']
            
            trace = Trace(data=traceData)
           
            
            
           

            trace.stats.station = self.sensors[sensorIndex - 1]
            trace.stats.sampling_rate = self.samplingRate
            trace.stats.channel = self.channel
            trace.stats.starttime = UTCDateTime(startTime) + start_sample * trace.stats.delta
            if trace.stats.starttime < self.global_starttime:
                self.global_starttime = trace.stats.starttime
            # print(trace.stats.station, trace.stats.starttime)
            stream.append(trace)

            del traceData  # Explicitly delete traceData to free memory
            del trace

            gc.collect()  # Force garbage collection

        return stream


    def _load_data_directory(self, directoryPath, num_files_directory=None):
        print('got inside load_dataDirectory')
        fileList = os.listdir(directoryPath)

        if num_files_directory is None:
            stream = Stream()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(self._load_data_txt, [os.path.join(directoryPath, fileName) for fileName in fileList if fileName.endswith('.txt')]))
            for result in results:
                stream.append(result)
            self.data_dict['1'] = stream
        elif isinstance(num_files_directory, int) and num_files_directory > 0:
            print('making multiple streams, events already have been split')
            for eventNum in range(1, num_files_directory + 1):
                iStream = Stream()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = list(executor.map(self._load_data_txt, [os.path.join(directoryPath, fileName) for fileName in fileList if fileName.endswith(f'event{str(eventNum).zfill(6)}.txt')]))
                for result in results:
                    iStream.append(result)
                self.data_dict[str(eventNum)] = iStream
        else:
            raise TypeError("Input num_files_directory must be an integer greater than zero")

    def apply_quakephase(self, parameters: str, useRegularPort: bool = False, isParallel: bool = False):
        """
        Apply ML picking module Quakephase to the loaded data.

        Parameters:
        parameters (str): Path to a .yaml file containing the parameters for applying Quakephase.
        useRegularPort (bool): Indication if port 22 should be used when calling a seisbench model from 
            inside quakephase. Should be used on a cluster first time Quakephase is run.


        Returns:
        - a dictionary containing a corresponding streams and traces to the self.data_dict, where each 
          trace represents probability of an AE event vs time

        Raises:
        - ValueError: If input is incorrect.
        """
        print('applying quakephase')
        if not parameters or not parameters.strip():
            raise ValueError("parameters cannot be an empty string.")
        if not os.path.isfile(parameters):
            raise ValueError("parameters must be a file")

        
        if isParallel:
            return self._apply_quakephase_parallel(parameters, useRegularPort)
        else:
            return self._apply_quakephase_sequential(parameters, useRegularPort)



    def _apply_quakephase_sequential(self, parameters, useRegularPort=False):
        """
        Apply ML picking module Quakephase to the loaded data. This method is intended 
        for internal use only and should only be called by the 'apply_quakephase' method.

        Parameters:
        parameters (str): Path to a .yaml file containing the parameters for applying Quakephase.
        useRegularPort (bool): Indication if port 22 should be used when calling a seisbench model from 
            inside quakephase. Should be used on a cluster first time Quakephase is run.


        Returns:
        - a dictionary containing a corresponding streams and traces to the self.data_dict, where each 
          trace represents probability of an AE event vs time

        Raises:
        - ValueError: If input is incorrect
        """
        if useRegularPort:
            seisbench.use_backup_repository()
        startTime = time.time()
        output_dictionary = {}
        
        for key in self.data_dict:
            startEventTime = time.time()
            print(f"applying quakephase to event {key}")
            sys.stdout.flush()
            output_dictionary[key] = quakephaseLBQ.apply(self.data_dict[key], parameters)
            print(f'time taken to apply quakephase to event {key}: {time.time() - startEventTime:.2f} seconds')
            sys.stdout.flush()

            self._clear_memory()
            
        elapsedTime = time.time() - startTime
        print(f'time taken to apply quakephase: {elapsedTime:.2f} seconds')
        sys.stdout.flush()
        return output_dictionary
    
    def _apply_quakephase_parallel(self, parameters, useRegularPort=False):
        """
        Apply ML picking module Quakephase to the loaded data in parallel with Dask. This method is intended 
        for internal use only and should only be called by the 'apply_quakephase' method.

        Parameters:
        parameters (str): Path to a .yaml file containing the parameters for applying Quakephase.
        useRegularPort (bool): Indication if port 22 should be used when calling a seisbench model from 
            inside quakephase. Should be used on a cluster first time Quakephase is run.


        Returns:
        - a dictionary containing a corresponding streams and traces to the self.data_dict, where each 
          trace represents probability of an AE event vs time

        Raises:
        - ValueError: If input is incorrect
        """
        if useRegularPort:
            seisbench.use_backup_repository()
        startTime = time.time()
        output_dictionary = {}
        
        for key in self.data_dict:
            startEventTime = time.time()
            print(f"applying quakephase to event {key}")
            sys.stdout.flush()
            output_dictionary[key] = quakephaseLBQParallel.apply(self.data_dict[key], parameters)
            print(f'time taken to apply quakephase to event {key}: {time.time() - startEventTime:.2f} seconds')
            sys.stdout.flush()

            self._clear_memory()
            
        elapsedTime = time.time() - startTime
        print(f'time taken to apply quakephase: {elapsedTime:.2f} seconds')
        sys.stdout.flush()
        return output_dictionary

    def _clear_memory(self):
        """
        Clear memory by invoking garbage collection and log memory usage.
        """
        gc.collect()
        memory_usage = self.get_memory_usage()
        print(f'Current memory usage: {memory_usage:.2f} MB')

    def get_memory_usage(self):
        """
        Returns the current local memory usage.
        
        Parameters:
        None
        
        Returns:
        mem_info: A string detailing the current memory usage within a computer system."""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 ** 2)
    
    def combine_streams(self, streams_dict):
        """
        Combines multiple ObsPy streams from a dictionary into a single stream. This method handles gaps between consecutive streams by filling
        them with zeros to ensure continuous data. It iterates over each sensor station and concatenates the data of traces with matching
        stations, handling time gaps appropriately.

        Parameters:
        - streams_dict (dict): Dictionary of batched streams returned from applying Quakephase.

        Returns:
        - combined_stream (Stream): A single obspy stream containing data which is the probability of an AE event at a given point in time,
        """
        combined_stream = Stream()

        # Iterate through each sensor
        for station in self.sensors:
            combined_data = np.array([])
            trace_prev = None

            # Iterate through each stream in the dictionary
            for key, stream in streams_dict.items():
                # Find traces with the current station
                for trace in stream['prob'].select(station=station, channel='prob_P'):
                    # Concatenate the data
                    if trace_prev is not None:
                        # Calculate the time difference between the end of the last trace and the start of the current trace
                        time_diff = trace.stats.starttime - trace_prev.stats.endtime
                        num_zeros_to_add = time_diff * self.samplingRate

                        if num_zeros_to_add > 0:
                            # If there's a gap, fill it with zeros
                            zero_array = np.zeros(int(num_zeros_to_add))
                            combined_data = np.concatenate((combined_data, zero_array))
                    else:
                        time_diff = trace.stats.starttime - self.global_starttime
                        num_zeros_to_add = time_diff * self.samplingRate

                        if num_zeros_to_add > 0:
                            # If there's a gap, fill it with zeros
                            zero_array = np.zeros(int(num_zeros_to_add))
                            combined_data = np.concatenate((combined_data, zero_array))

                    # Concatenate the current trace's data
                    combined_data = np.concatenate((combined_data, trace.data))
                    trace_prev = trace

            # Create a new trace with the combined data
            if combined_data.size > 0:

                new_trace = Trace(data=combined_data)
                # Set stats for the new trace
                stats = streams_dict['0']['prob'].select(station=station, channel='prob_P')[0].stats
                new_trace.stats.station = stats.station
                new_trace.stats.starttime = stats.starttime
                new_trace.stats.channel = stats.channel
                new_trace.stats.sampling_rate = stats.sampling_rate

                del stats


                # Add the new trace to the combined stream
                combined_stream.append(new_trace)

        return combined_stream




class AcousticEmissionWrapperController:
    def __init__(self, acousticEmissionWrapper):
        self.acousticEmissionWrapper = acousticEmissionWrapper

    def load_data(self, path=None, directoryName=False, dataStream=None, num_files_directory=None):
        self.acousticEmissionWrapper.load_data(path, directoryName, dataStream, num_files_directory)

    def apply_quakephase(self, parameters, useRegularPort=False, parallelProcessing=False, parallelStreams=None, maxWorkers=None):
        return self.acousticEmissionWrapper.apply_quakephase(parameters, useRegularPort, parallelProcessing, parallelStreams, maxWorkers)

    def save_data(self, quakePhaseOutput=None, saveMethod='pickle', fileName='acousticEmissionWrapper'):
        self.acousticEmissionWrapper.saveData(quakePhaseOutput, saveMethod, fileName)

    def get_memory_usage(self):
        return self.acousticEmissionWrapper.get_memory_usage()

def average_traces(stream):
    """
    Averages all trace's """
    if not isinstance(stream, Stream):
        raise TypeError("Input must be an ObsPy Stream object")
    
    num_traces = len(stream)
    if num_traces == 0:
        raise ValueError("Stream contains no traces")
    
    trace_length = len(stream[0].data)
    for trace in stream:
        if len(trace.data) != trace_length:
            raise ValueError("All traces in the stream must have the same length")

    stacked_data = np.zeros(trace_length)
    for trace in stream:
        stacked_data += trace.data
    print(stream[0].stats.starttime)
    averaged_data = stacked_data / num_traces
    averaged_trace = Trace(data=averaged_data)
    averaged_trace.stats = stream[0].stats.copy()
    averaged_trace.stats.station = 'avg'
    return averaged_trace

def multiply_traces(stream):
    multiplied_trace = Trace()

    trace_length = len(stream[0].data)
    for trace in stream:
        if len(trace.data) != trace_length:
            raise ValueError("All traces in the stream must have the same length")


    multiplied_data = np.full((1,trace_length),1, dtype=float)[0,:]

    for trace in stream:
        multiplied_data *= trace.data
    print(stream[0].stats.starttime)
    multiplied_trace = Trace(data=multiplied_data)
    multiplied_trace.stats = stream[0].stats.copy()
    multiplied_trace.stats.station = 'multiplied'

    return multiplied_trace 


def plot_pwave_probabilities(stream, station_name):
    # Find the correct trace based on station name
    trace = None
    for tr in stream:
        if tr.stats.station == station_name:
            trace = tr
            break

    if trace is None:
        print(f"No trace found with station name '{station_name}'")
        return

    # Extract data and time
    data = trace.data
    times = np.linspace(0, trace.stats.endtime - trace.stats.starttime, num=len(data))

    # Find peaks
    min_peak_height = 0.1
    min_peak_distance = int(0.005 / trace.stats.delta)
    peaks, _ = find_peaks(data, height=min_peak_height, distance=min_peak_distance)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(times, data, 'k-', label='P-Wave Probability')
    plt.scatter(times[peaks], data[peaks], color='red', label='Highest Peaks', zorder=3)
    
    plt.yscale('log')
    plt.xlabel('Relative Time')
    plt.ylabel('P-Wave Probability')
    plt.title(f'P-Wave Probability with Highest Peaks in Time Windows for Event {trace.stats.network}{trace.stats.station}')
    plt.legend()

    # Customize grid lines to appear only at powers of 10
   

    plt.show()


