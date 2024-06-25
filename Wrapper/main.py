import cProfile
import pstats
import psutil
import timeit
import os
from Catalogue import Catalogue, average_traces, multiply_traces
import gc
import pickle

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
    newCatalogue = Catalogue()

    # Load data
    newCatalogue.loadData(path=path, isDataFile=True, num_batches=10)

    # Apply quakephase
    output = newCatalogue.applyQuakephase(parameters=parameters)

    # Save results
    combined_stream = newCatalogue.combine_streams(output)
    del output
    gc.collect()
    total_out = {}
    total_out['avg'] = average_traces(combined_stream)
    total_out['multiplied'] = multiply_traces(combined_stream)
    del combined_stream
    gc.collect()

    with open('/cluster/scratch/nmunro/onlyPhaseNet_100kHz.pickle', 'wb') as file:
        pickle.dump(combined_stream, file)
def main():
    parameters = '/cluster/scratch/nmunro/quakephaseNoah/Wrapper/parameters.yaml'
    path = '/cluster/scratch/nmunro/tpc5File/LBQ-20220331-I-BESND_086a.tpc5'
    run_quakephase(parameters, path)

if __name__ == "__main__":
    main()
