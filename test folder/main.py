import cProfile
import pstats
import psutil
import timeit
import os
from Catalogue import AcousticEmissionWrapper, average_traces, multiply_traces
import gc
import pickle
import scipy.io

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
    sensors = ['A1', 'A3', 'B1', 'B3', 'C1', 'C3', 'D1', 'D3', 'E1', 'E3', 'F1', 'F3', 'G1', 'G3', 'H1', 'H3']
    wrapper = AcousticEmissionWrapper(sensors = sensors, samplingRate = 10e6, channel = 'FPZ')

    # Load data
    wrapper.load_data(path=path, num_batches=20)

    # Apply quakephase
    output = wrapper.apply_quakephase(parameters=parameters)

    # Save results
    combined_stream = wrapper.combine_streams(output)
    del output
    gc.collect()
    total_out = {}
    total_out['avg'] = average_traces(combined_stream)
    total_out['multiplied'] = multiply_traces(combined_stream)
    del combined_stream
    gc.collect()

    with open('/cluster/scratch/nmunro/testNewWrapper.pickle', 'wb') as file:
        pickle.dump(total_out, file)
        
    #save to mat file
    mat_dict = {'multiplied': [total_out['multiplied'].data, total_out['multiplied'].stats.starttime.timestamp], 'avg': [total_out['avg'].data, total_out['avg'].stats.starttime.timestamp]}
    
    # Save the dictionary to a .mat file
    scipy.io.savemat('/cluster/scratch/nmunro/testNewWrapper.mat', mat_dict)

def main():
    parameters = '/cluster/scratch/nmunro/quakephaseNoah/Wrapper/parameters.yaml'
    path = '/cluster/scratch/nmunro/tpc5File/LBQ-20220331-I-BESND_086a.tpc5'
    run_quakephase(parameters, path)

if __name__ == "__main__":
    main()
