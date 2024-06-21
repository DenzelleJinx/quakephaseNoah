import concurrent.futures
import os
import numpy as np
import h5py
import pickle
import time
import pandas as pd
import sys
import psutil
from obspy.core import Stream, Trace
from quakephase import quakephase
import seisbench
import gc
from multiprocessing import shared_memory


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
        if hdf5File is None:
            with h5py.File(path, 'r') as hdf5File:
                stream = self._process_tpc5_file(hdf5File, num_samples_per_batch, batch_index)
        else:
            stream = self._process_tpc5_file(hdf5File, num_samples_per_batch, batch_index)

        # Convert stream to a format suitable for shared memory
        for trace in stream:
            trace_data = trace.data
            shm = shared_memory.SharedMemory(create=True, size=trace_data.nbytes)
            shm_array = np.ndarray(trace_data.shape, dtype=trace_data.dtype, buffer=shm.buf)
            np.copyto(shm_array, trace_data)
            trace.shared_memory_name = shm.name  # Store shared memory name in the trace for later use

        return stream

    def _process_tpc5_file(self, hdf5File, num_samples_per_batch, batch_index):
        stream = Stream()
        channelPathTemplate = '/measurements/00000001/channels/{:08d}/blocks/00000001/data'
        if batch_index is None:
            batch_index = 0

        start_sample = batch_index * num_samples_per_batch
        end_sample = min((batch_index + 1) * num_samples_per_batch, hdf5File[channelPathTemplate.format(1)].shape[0])

        for sensorIndex in range(1, len(self.sensors) + 1):
            channelPath = channelPathTemplate.format(sensorIndex)
            traceData = np.array(hdf5File[channelPath][start_sample:end_sample])  # Load batch of data
            trace = Trace(data=traceData)
            trace.stats.station = self.sensors[sensorIndex - 1]
            trace.stats.sampling_rate = self.samplingRate
            trace.stats.channel = self.channel
            stream.append(trace)
            del traceData

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

    def applyQuakephase(self, parameters, useRegularPort=False, parallelProcessing=False, maxWorkers=None):
        print('applying quakephase')
        if parallelProcessing:
            return self._apply_quakephase_parallel(parameters, useRegularPort, maxWorkers)
        else:
            return self._apply_quakephase_sequential(parameters, useRegularPort)

    def _apply_quakephase_parallel(self, parameters, useRegularPort=False, maxWorkers=None):
        print('runningparallel')
        if useRegularPort:
            seisbench.use_backup_repository()
        startTime = time.time()
        outputDictionary = {}

        eventKeys = list(self.events.keys())
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=maxWorkers) as executor:
            args = [(key, self.events[key], parameters) for key in eventKeys]
            results = list(executor.map(self.applyToEventHelper, args))
        
        for key, result in results:
            outputDictionary[key] = result
        print(f"Memory usage after processing: {self.get_memory_usage():.2f} MB")
        
        elapsedTime = time.time() - startTime
        print(f'time taken to apply quakephase: {elapsedTime:.2f} seconds')
        del results
        self._clear_memory()
        return outputDictionary






    def _apply_quakephase_sequential(self, parameters, useRegularPort=False):

        if useRegularPort:
            seisbench.use_backup_repository()
        startTime = time.time()
        outputDictionary = {}
        
        for idx, key in enumerate(self.events):
            for trace in self.events[key]:
                print(f"Original trace length: {len(trace.data)}, Memory usage: {sys.getsizeof(trace.data) / (1024 ** 2):.2f} MB")
    
            startEventTime = time.time()
            outputDictionary[key] = quakephase.apply(self.events[key], parameters)
            print(f'time taken to apply quakephase to event {key}: {time.time() - startEventTime:.2f} seconds')

            if idx % 10 == 0:
                print(f"Memory usage after processing {idx + 1} events: {self.get_memory_usage():.2f} MB")
            self._clear_memory()
            
        elapsedTime = time.time() - startTime
        print(f'time taken to apply quakephase: {elapsedTime:.2f} seconds')
        return outputDictionary

    def applyToEvent(self, eventKey, eventData, parameters):
        # Access shared memory in eventData
        for trace in eventData:
            shm = shared_memory.SharedMemory(name=trace.shared_memory_name)
            trace_data = np.ndarray(trace.data.shape, dtype=trace.data.dtype, buffer=shm.buf)
            trace.data = trace_data  # Reassign data from shared memory

        result = quakephase.apply(eventData, parameters)
        
        # Cleanup shared memory
        for trace in eventData:
            shm.close()
            shm.unlink()
        
        return eventKey, result

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
    return averaged_trace

def test_apply_quakephase():
    catalogue = Catalogue()
    controller = CatalogueController(catalogue)
    # Load a small test dataset
    controller.load_data(path='/Users/noahmunro-kagan/Downloads/Quakephase related scripts/Quakephase Test Data/Data', directoryName=True, numberOfEvents=2)
    parameters = '/Users/noahmunro-kagan/Downloads/Quakephase related scripts/parameters.yaml'
    output = controller.apply_quakephase(parameters=parameters, parallelProcessing=True)
    assert output is not None
    print("Test passed.")

# test_apply_quakephase()

path = '/Users/noahmunro-kagan/Desktop/Sample_data_Noah'

def test_return_loaded_data():
    catalogue = Catalogue()
    data = catalogue._load_data_tpc5(path)
    assert data is not None
    print("Data returned successfully.")

def test_return_loaded_data_batched():
    catalogue = Catalogue()
    batch_duration = 5  # Duration in seconds for each batch
    data = catalogue._load_data_tpc5(path, batch_duration=batch_duration)
    assert data is not None
    print("Data returned in batches successfully.")


import unittest
import time

class TestCatalogue(unittest.TestCase):

    def setUp(self):
        self.catalogue = Catalogue()
        self.path = '/Users/noahmunro-kagan/Desktop/Sample_data_Noah/LBQ-20220331-I-BESND_286.tpc5'  # Update this path to your test .tpc5 file

    def test_load_data_tpc5_full(self):
        start_time = time.time()
        self.catalogue.loadData(path=self.path, isDataFile=True)
        end_time = time.time()
        print(f"Time taken to load full data: {end_time - start_time:.2f} seconds")

        self.assertTrue(len(self.catalogue.events) > 0, "No data loaded")
        fileNumber = list(self.catalogue.events.keys())[0]
        self.assertIn('full', self.catalogue.events[fileNumber], "Full data not loaded correctly")
        print(self.catalogue.events['286']['full'])
        print("Test load full data passed.")

    def test_load_data_tpc5_batched(self):
        num_batches = 5  # Number of batches to divide the data into
        start_time = time.time()
        self.catalogue.loadData(path=self.path, isDataFile=True, num_batches=num_batches)
        end_time = time.time()
        print(f"Time taken to load data in batches: {end_time - start_time:.2f} seconds")

        self.assertTrue(len(self.catalogue.events) > 0, "No data loaded")
        fileNumber = list(self.catalogue.events.keys())[0]
        self.assertEqual(len(self.catalogue.events[fileNumber]), num_batches, "Data not loaded into correct number of batches")
        print("Test load batched data passed.")
        print(self.catalogue.events['286']['0'])

    def tearDown(self):
        del self.catalogue
        gc.collect()

if __name__ == '__main__':
    unittest.main()


