import time

def get_avg_time(runs, func, *args, **kwargs):

    avg = 0
    runs = runs
    for i in range(100):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_elapsed = end_time - start_time
    for i in range(runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        time_elapsed = end_time - start_time
        avg_t = time_elapsed
        avg += avg_t

    print(f"Avg Time taken: {avg/runs:.4f} seconds")
    return result, avg/runs