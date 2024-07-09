import cProfile
import pstats
import psutil
import timeit
import os
from Catalogue import AcousticEmissionWrapper, average_traces, multiply_traces
import gc
import pickle
import scipy.io
from scipy.signal import find_peaks
import numpy as np
import time

def profile_code(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        result = func(*args, **kwargs)
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()
        return result
    return wrapper

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - start_time
        print(f"Time taken for {func.__name__}: {elapsed:.2f} seconds")
        return result
    return wrapper

def monitor_resources(func):
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        before_cpu = process.cpu_percent(interval=None)
        before_mem = process.memory_info().rss / 1024 ** 2

        result = func(*args, **kwargs)

        after_cpu = process.cpu_percent(interval=None)
        after_mem = process.memory_info().rss / 1024 ** 2

        print(f"CPU usage for {func.__name__}: {after_cpu - before_cpu:.2f}%")
        print(f"Memory usage for {func.__name__}: {after_mem - before_mem:.2f} MB")
        return result
    return wrapper

@profile_code
@measure_time
@monitor_resources
def run_quakephase(parameters, path):
    # Initialize the Catalogue instance
    startTime = time.time()
    def find_max_and_index_numpy(arr):
        if arr.size == 0:  # Check if the array is empty
            return None, None
        max_value = np.max(arr)
        max_index = np.argmax(arr)
        return max_value, max_index


    sensors = ['A1', 'A3', 'B1', 'B3', 'C1', 'C3', 'D1', 'D3', 'E1', 'E3', 'F1', 'F3', 'G1', 'G3', 'H1', 'H3']
    wrapper = AcousticEmissionWrapper(sensors = sensors, samplingRate = 10e6, channel = 'FPZ')

    # Load data
    wrapper.load_data(path=path, num_batches=20)

    # Apply quakephase
    output = wrapper.apply_quakephase(parameters=parameters, isParallel=True)

    # Save results
    combined_stream = wrapper.combine_streams(output)
    del output
    gc.collect()
    total_out = {}
    total_out['avg'] = average_traces(combined_stream)
    total_out['multiplied'] = multiply_traces(combined_stream)
    total_out['all'] = combined_stream
    gc.collect()

    # with open('/cluster/scratch/nmunro/allDotsnodask.pickle', 'wb') as file:
    #     pickle.dump(total_out, file)
        
    #save to mat file
    mat_dict = {'multiplied': [total_out['multiplied'], total_out['multiplied'].stats.starttime.timestamp], 'avg': [total_out['avg'], total_out['avg'].stats.starttime.timestamp]}
    
    mat_dict['raw'] = {}

    mul_peaks, _ = find_peaks(total_out['multiplied'][0].data, height = int(10e-60), distance = int(0.005 * wrapper.samplingRate))
    avg_peaks, _ = find_peaks(total_out['avg'][0].data, height = 10e-60, distance = int(0.005 * wrapper.samplingRate))

    mat_dict['raw']['mul_peaks'] = {}

    for peak_index in mul_peaks:
            
        for sensor in wrapper.getSensors():
            
            mat_dict['raw']['mul_peaks'][sensor] = {}
            data = combined_stream.select(station=sensor)[0].data[peak_index - 1000: peak_index + 1000]

            i_max_value, i_max_value_index = find_max_and_index_numpy(data) 



            mat_dict['raw']['mul_peaks'][sensor]['max_index'] = i_max_value_index
            mat_dict['raw']['mul_peaks'][sensor]['max_probability'] = i_max_value
            del data
            gc.collect()


        mat_dict['raw']['avg_peaks'] = {}
        
        for peak_index in avg_peaks:
                
            for sensor in wrapper.getSensors():
                
                mat_dict['raw']['avg_peaks'][sensor] = {}
                data = combined_stream.select(station=sensor)[0].data[peak_index - 1000: peak_index + 1000]

                i_max_value, i_max_value_index = find_max_and_index_numpy(data)



                mat_dict['raw']['avg_peaks'][sensor]['max_index'] = i_max_value_index
                mat_dict['raw']['avg_peaks'][sensor]['max_probability'] = i_max_value
                del data
                gc.collect()


    # Save the dictionary to a .mat file
    scipy.io.savemat('/cluster/scratch/nmunro/chatgpt.mat', mat_dict)
    print(len(total_out['multiplied'].data), len(total_out['avg'].data))
    endTime = time.time()
    print(endTime - startTime)

def main():
    parameters = '/cluster/scratch/nmunro/quakephaseNoah/Wrapper/parameters.yaml'
    path = '/cluster/scratch/nmunro/tpc5File/LBQ-20220331-I-BESND_086a.tpc5'
    run_quakephase(parameters, path)

if __name__ == "__main__":
    main()
