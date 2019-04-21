import csv
import json
import os
import statistics
import time
from datetime import datetime

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


working_dir = os.path.dirname(os.path.realpath(__file__))
throughput_dir = os.path.join(working_dir, 'throughput_data')

stats_filename = 'stats.csv'
timeseries_filename = 'stats_raw.csv'
timeseries_plot = 'time_series.png'
ignore_set = {stats_filename, timeseries_filename, timeseries_plot}


def _plot_time_series(time_series_stats):
    times = []
    counts = []
    for stat in time_series_stats:
        times.append(stat['timestamp'])
        counts.append(stat['car_count'])
    if plt:
        plt.plot(times, counts, 'bo-')
        plt.xlabel('Time [Unix]')
        plt.ylabel('Number of Cars')
        plt.title('Car Count over Time')
        plt.savefig(os.path.join(throughput_dir, timeseries_plot))
        # plt.show()  # needs hacks to work with Docker


def summarize_throughput():
    # track overall stats
    stats_path = os.path.join(throughput_dir, stats_filename)

    car_counts = []
    time_series_stats = []
    for file in os.listdir(throughput_dir):
        if file in ignore_set:
            continue  # skip the aggregate files
        with open(os.path.join(throughput_dir, file)) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                car_count = int(row['car_count'])
                car_counts.append(car_count)
                time_series_stats.append({
                    'car_count': car_count,
                    'filename': file,
                    'timestamp': row['timestamp'],
                    'predictions_save_path': row['predictions_save_path'],
                })
    time_series_stats.sort(key=lambda x: x['filename'])
    overall_stats = {
        'max_count': max(car_counts),
        'min_count': min(car_counts),
        'mean_count': statistics.mean(car_counts),
        'median_count': statistics.median(car_counts),
        'standard_deviation': statistics.stdev(car_counts),
    }

    print('Time series on throughput')
    print(json.dumps(time_series_stats, indent=4))
    _plot_time_series(time_series_stats)
    with open(os.path.join(throughput_dir, timeseries_filename), 'w') as csvfile:
        fieldnames = time_series_stats[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for stat in time_series_stats:
            writer.writerow(stat)

    print('\nSummary statistics on throughput')
    print(json.dumps(overall_stats, indent=4))

    with open(stats_path, 'w') as csvfile:
        fieldnames = overall_stats.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(overall_stats)

    return overall_stats


def track_throughput(boxes, predictions_save_path):
    # track the specific image stats
    base = os.path.splitext(os.path.basename(predictions_save_path))[0]
    unix_time = time.time()
    logs_path = os.path.join(throughput_dir, f'{base}.csv')

    boxes_stats = {
        'car_count': len(boxes),
        'timestamp': datetime.fromtimestamp(unix_time).isoformat(),
        'predictions_save_path': predictions_save_path,
    }
    with open(logs_path, 'w') as csvfile:
        fieldnames = boxes_stats.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(boxes_stats)

    return boxes_stats


if __name__ == '__main__':
    summarize_throughput()
