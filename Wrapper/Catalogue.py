import concurrent.futures
import os
import numpy as np
import h5py
import pickle
import time
import pandas as pd
import sys
import psutil
from obspy.core import Stream, Trace, UTCDateTime
from dask.distributed import Client, LocalCluster, as_completed
import dask
from quakephase import quakephase

import seisbench
import gc

#plotting imports
import matplotlib.pyplot as plt

from obspy import read
from scipy.signal import find_peaks

# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# # Add the parent directory to the system path
# sys.path.append(parent_dir)

# # Now you can import the quakephase module
# from quakephaseNoah import quakephase1

class Catalogue:
    def __init__(self, sensors=None, samplingRate=None, channel=None):
        self.sensors = sensors or ['A1', 'A3', 'B1', 'B3', 'C1', 'C3', 'D1', 'D3', 'E1', 'E3', 'F1', 'F3', 'G1', 'G3', 'H1', 'H3']
        self.samplingRate = samplingRate or 10e6
        self.channel = channel or 'FPZ'
        self.events = {}

    def loadData(self, path=None, isDirectory=False, isDataFile=False, dataStream=None, numberOfEvents=None, num_batches=None):
        if (sum(x is not None for x in [path, dataStream]) != 1):
            raise ValueError("Exactly one of path or dataStream must be provided")
        if path is not None and (sum([isDirectory, isDataFile]) != 1):
            raise ValueError("Input path cannot be for a directory and datafile")

        if (path is not None) and isDataFile:
            self._load_data_file(path=path, num_batches=num_batches)
        elif path is not None and isDirectory:
            self._load_data_directory(path, numberOfEvents)
        elif dataStream is not None:
            self._load_data_stream(dataStream)
        else:
            raise SyntaxError("Unknown error occurred, please check your input for loadData")


    def _load_data_file(self, path, num_batches=None):
        print('loading data')
        fileEnding = path.split('.')[-1]
        if fileEnding == 'txt':
            return self._load_data_txt(path)
        elif fileEnding == 'mat':
            self._load_data_mat(path)
        elif fileEnding == 'tpc5':
            self._load_data_tpc5(path=path, num_batches=num_batches)
        else:
            raise TypeError("Input file must be .mat, .txt or .tpc5 file")

        
    def _load_data_txt(self, path):
        """
        loads data from a txt file to a single trace
        """
        fileBaseName = os.path.basename(path)
        fileBaseNameSplit = fileBaseName.split('_')
        eventNumber = int((fileBaseNameSplit[-1].split('.')[0].lstrip('event')).lstrip('0'))
        sensorName = fileBaseNameSplit[-2]

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
        print('got inside loadDataMat, implementation needed')
        # Implementation needed

    def _load_data_tpc5(self, path, num_batches=None):
        """
        Load data from a .tpc5 file, optionally in batches, to manage memory efficiently.

        Parameters:
        path (str): Path to the .tpc5 file
        num_batches (int, optional): Number of batches to divide the data into. Defaults to None (no batching).
        """
        print('Loading data from .tpc5 file')
        if num_batches is None:
            self._load_data_tpc5_full(path)
        else:
            self._load_data_tpc5_batched(path=path, num_batches=num_batches)

    def _load_data_tpc5_full(self, path):
        """
        Load the entire .tpc5 file at once into a single stream.

        Parameters:
        path (str): Path to the .tpc5 file
        """
        with h5py.File(path, 'r') as hdf5File:
            channelPathTemplate = '/measurements/00000001/channels/{:08d}/blocks/00000001/data'

            # Determine the total number of samples (assume all sensors have the same number of samples)
            total_samples = hdf5File[channelPathTemplate.format(1)].shape[0]

        stream = self._load_data_tpc5_internal(path=path, num_samples_per_batch=total_samples)
        fileBaseName = os.path.basename(path)
        fileNumber = ((fileBaseName.split('_')[-1]).split('.')[0]).lstrip('tpc5')
        self.events['full'] = stream
        print(f'Loaded data for event {fileNumber} from .tpc5 file')

        del stream
        self._clear_memory()

    def _load_data_tpc5_batched(self, path, num_batches):
        """
        Load the .tpc5 file in batches to conserve memory.

        Parameters:
        path (str): Path to the .tpc5 file
        num_batches (int): Number of batches to divide the data into.
        """
        print(f'loading tpc5 file in {num_batches} batches')
        fileBaseName = os.path.basename(path)
        fileNumber = ((fileBaseName.split('_')[-1]).split('.')[0]).lstrip('tpc5')
        
        with h5py.File(path, 'r') as hdf5File:
            channelPathTemplate = '/measurements/00000001/channels/{:08d}/blocks/00000001/data'

            # Determine the total number of samples (assume all sensors have the same number of samples)
            total_samples = hdf5File[channelPathTemplate.format(1)].shape[0]
            num_samples_per_batch = total_samples // num_batches  # integer division

            for batch_index in range(num_batches):
                stream = self._load_data_tpc5_internal(path, num_samples_per_batch=num_samples_per_batch, hdf5File=hdf5File, batch_index=batch_index)
                self.events[str(batch_index)] = stream
                print(f'Loaded batch {batch_index + 1}/{num_batches} for event {fileNumber}')
                del stream
                self._clear_memory()

    def _load_data_tpc5_internal(self, path, num_samples_per_batch, hdf5File=None, batch_index=None):
        try:
            print(f"Loading data batch {batch_index} from {path}...")
            if hdf5File is None:
                with h5py.File(path, 'r') as hdf5File:
                    stream = self._process_tpc5_file(hdf5File, num_samples_per_batch, batch_index)
            else:
                stream = self._process_tpc5_file(hdf5File, num_samples_per_batch, batch_index)

            print(f"Data batch {batch_index} loaded successfully")
            return stream
        except Exception as e:
            print(f"Error loading data for batch {batch_index}: {e}")
            return None





    def _process_tpc5_file(self, hdf5File, num_samples_per_batch, batch_index):
        stream = Stream()
        channelPathTemplate = '/measurements/00000001/channels/{:08d}/blocks/00000001/data'
        if batch_index is None:
            batch_index = 0

        start_sample = batch_index * num_samples_per_batch
        end_sample = min((batch_index + 1) * num_samples_per_batch, hdf5File[channelPathTemplate.format(1)].shape[0])
        
        for sensorIndex in range(1, len(self.sensors) + 1):
            channelPath = channelPathTemplate.format(sensorIndex)
            channelPathTime = channelPath.strip('data')
            traceData = np.array(hdf5File[channelPath][start_sample:end_sample])
            print(type(traceData))
            startTime = hdf5File[channelPathTime].attrs['startTime']
            
            trace = Trace(data=traceData)
            trace.stats.starttime = UTCDateTime(startTime)
            trace.stats.station = self.sensors[sensorIndex - 1]
            trace.stats.sampling_rate = self.samplingRate
            trace.stats.channel = self.channel
            stream.append(trace)
            del traceData  # Explicitly delete traceData to free memory
            del trace

            gc.collect()  # Force garbage collection

        return stream



    def _clear_memory(self):
        """
        Clear memory by invoking garbage collection and log memory usage.
        """
        gc.collect()
        memory_usage = self.get_memory_usage()
        print(f'Current memory usage: {memory_usage:.2f} MB')


    def _load_data_directory(self, directoryPath, numberOfEvents=None):
        print('got inside loadDataDirectory')
        fileList = os.listdir(directoryPath)

        if numberOfEvents is None:
            stream = Stream()
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(executor.map(self._load_data_txt, [os.path.join(directoryPath, fileName) for fileName in fileList if fileName.endswith('.txt')]))
            for result in results:
                stream.append(result)
            self.events['1'] = stream
        elif isinstance(numberOfEvents, int) and numberOfEvents > 0:
            print('making multiple streams, events already have been split')
            for eventNum in range(1, numberOfEvents + 1):
                iStream = Stream()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    results = list(executor.map(self._load_data_txt, [os.path.join(directoryPath, fileName) for fileName in fileList if fileName.endswith(f'event{str(eventNum).zfill(6)}.txt')]))
                for result in results:
                    iStream.append(result)
                self.events[str(eventNum)] = iStream
        else:
            raise TypeError("Input numberOfEvents must be an integer greater than zero")

    def applyQuakephase(self, parameters, useRegularPort=False, parallelProcessing=False):
        print('applying quakephase')
        if parallelProcessing:
            return self._apply_quakephase_parallel(parameters, useRegularPort)
        else:
            return self._apply_quakephase_sequential(parameters, useRegularPort)

    def _apply_quakephase_parallel(self, parameters, useRegularPort=False):
        print('running parallel')
        client = Client()  # This ensures tasks are distributed across available cores

        if useRegularPort:
            seisbench.use_backup_repository()
        startTime = time.time()

        # Creating delayed tasks
        futures = {key: dask.delayed(quakephase.apply)(stream, parameters) for key, stream in self.events.items()}

        # Computing results in parallel
        results = dask.compute(*futures.values())

        outputDictionary = {key: result for key, result in zip(futures.keys(), results)}

        print(f"Memory usage after processing: {self.get_memory_usage():.2f} MB")
        elapsedTime = time.time() - startTime
        print(f'time taken to apply quakephase: {elapsedTime:.2f} seconds')
        self._clear_memory()
        return outputDictionary


    def _apply_quakephase_sequential(self, parameters, useRegularPort=False):
        if useRegularPort:
            seisbench.use_backup_repository()
        startTime = time.time()
        outputDictionary = {}
        
        for key in self.events:
            startEventTime = time.time()
            print(f"applying quakephase to event {key}")
            sys.stdout.flush()
            outputDictionary[key] = quakephase.apply(self.events[key], parameters)
            print(f'time taken to apply quakephase to event {key}: {time.time() - startEventTime:.2f} seconds')
            sys.stdout.flush()

            self._clear_memory()
            
        elapsedTime = time.time() - startTime
        print(f'time taken to apply quakephase: {elapsedTime:.2f} seconds')
        sys.stdout.flush()
        return outputDictionary

    def applyToEvent(self, eventKey, eventData, parameters):
        try:
            print(f"Processing event {eventKey}...")

            result = quakephase.apply(eventData, parameters)
            print(f"Quakephase applied for event {eventKey}")

            return eventKey, result
        except Exception as e:
            print(f"Error processing event {eventKey}: {e}")
            return eventKey, None






    def applyToEventHelper(self, args):
        eventKey, eventData, parameters = args
        return self.applyToEvent(eventKey, eventData, parameters)




    def saveData(self, quakePhaseOutput=None, saveMethod='pickle', fileName='Catalogue'):
        # savingDictionary = {'rawData': self.events}
        # if quakePhaseOutput is not None:
        #     savingDictionary['output'] = quakePhaseOutput
        if saveMethod == 'pickle':
            with open(f'{fileName}.pickle', 'wb') as file:
                pickle.dump(quakePhaseOutput, file)
        del self.events
        self._clear_memory()
        print(f'Data saved as {saveMethod} file')

    def get_memory_usage(self):
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return mem_info.rss / (1024 ** 2)
    
    def combine_streams(self, streams_dict):
        """
        Method which combines multiple streams into one
        """
        combined_stream = Stream()

        # Iterate through each sensor
        for station in self.sensors:
            combined_data = np.array([])

            # Iterate through each stream in the dictionary
            for key, stream in streams_dict.items():
                # Find traces with the current station
                for trace in stream['prob'].select(station=station, channel='prob_P'):
                    # Concatenate the data
                    print(np.shape(trace.data))
                    combined_data = np.concatenate((combined_data, trace.data))

            # Create a new trace with the combined data
            if combined_data.size > 0:
                new_trace = Trace(data=combined_data)
                # Set the station and other stats from the first trace (if needed)

                new_trace.stats.station = station





                # Add the new trace to the combined stream
                combined_stream.append(new_trace)

        return combined_stream




class CatalogueController:
    def __init__(self, catalogue):
        self.catalogue = catalogue

    def load_data(self, path=None, directoryName=False, dataStream=None, numberOfEvents=None):
        self.catalogue.loadData(path, directoryName, dataStream, numberOfEvents)

    def apply_quakephase(self, parameters, useRegularPort=False, parallelProcessing=False, parallelStreams=None, maxWorkers=None):
        return self.catalogue.applyQuakephase(parameters, useRegularPort, parallelProcessing, parallelStreams, maxWorkers)

    def save_data(self, quakePhaseOutput=None, saveMethod='pickle', fileName='Catalogue'):
        self.catalogue.saveData(quakePhaseOutput, saveMethod, fileName)

    def get_memory_usage(self):
        return self.catalogue.get_memory_usage()

def average_traces(stream):
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
    
    averaged_data = stacked_data / num_traces
    averaged_trace = Trace(data=averaged_data)
    averaged_trace.stats = stream[0].stats.copy()
    averaged_trace.stats.station = 'avg'
    return averaged_trace

def multiply_traces(stream):
    multiplied_trace = Stream()

    trace_length = len(stream[0].data)
    for trace in stream:
        if len(trace.data) != trace_length:
            raise ValueError("All traces in the stream must have the same length")


    multiplied_data = np.full((1,trace_length),1, dtype=float)[0,:]

    for trace in stream:
        multiplied_data *= trace.data

    multiplied_trace = Trace(data=multiplied_data)
    multiplied_trace.stats = stream[0].stats.copy()
    multiplied_trace.stats.station = 'multiplied'

    return multiplied_trace 

# with open('/Users/noahmunro-kagan/Desktop/Quakephase Outputs/batched10ParallelA/batched10Parallelaoutput.pickle', 'rb') as f:
#     loaded_dict = pickle.load(f)

# catalogue = Catalogue()
# combined_stream = catalogue.compile_batches(loaded_dict)
# mul_stream = multiply_traces(combined_stream)
# avg_stream = average_traces(combined_stream)
# combined_stream += mul_stream
# combined_stream += avg_stream
# print(mul_stream, avg_stream)

# combined_stream.write('/Users/noahmunro-kagan/Desktop/Quakephase Outputs/batched10ParallelA/phasenetandallfreq.sac', format='SAC')
# print(avg_stream[0].data)


# Function to plot P-Wave probability with peaks



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


# # Example usage
# # Replace 'path_to_stream_file' with the path to your ObsPy stream file
# stream = read('/Users/noahmunro-kagan/Desktop/Quakephase Outputs/batched10ParallelA/phasenetandallfreq18.sac')
# print(stream)
# station_name = 'avg'  # or 'avg'
# plot_pwave_probabilities(stream, station_name)

# newCatalogue = Catalogue()
# path = '/Users/noahmunro-kagan/Desktop/Quakephase Outputs/phaseNet+EQT_70+100kHz+10MHz/phaseNet+EQT_70+100kHz+10MHz_10batches_25minsPerSecond.pickle'

# with open(path, 'rb') as file:
#     loaded_dict = pickle.load(file)

# stream = newCatalogue.combine_streams(loaded_dict)
# # print(stream)
# # print(stream[0])
# avg_stream = average_traces(stream)
# # print(stream)
# # print(stream[0])
# mul_stream = multiply_traces(stream)
# # print(stream)
# # print(stream[0])

# total_dict = {'avg': avg_stream, 'multiplied': mul_stream}

# pathdump='/Users/noahmunro-kagan/Downloads/QuakephaseOutput/phaseNet+EQT_70+100kHz+10MHz_10batches_25minsPerSecond.pickle'

# with open(pathdump, 'wb') as f:
#     pickle.dump(total_dict, f)