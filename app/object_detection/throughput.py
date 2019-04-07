import csv
import os
import statistics
import subprocess
import time
from datetime import datetime


working_dir = os.path.dirname(os.path.realpath(__file__))
throughput_dir = os.path.join(working_dir, 'throughput')


def summarize_throughput():
    # track overall stats
    stats_filename = 'stats.csv'
    stats_path = os.path.join(throughput_dir, stats_filename)

    car_counts = []
    for file in os.listdir(throughput_dir):
        if file == stats_filename:
            continue  # skip the aggregated stats file
        with open(os.path.join(throughput_dir, file)) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                car_count = int(row['car_count'])
                car_counts.append(car_count)
    overall_stats = {
        'max_count': max(car_counts),
        'min_count': min(car_counts),
        'mean_count': statistics.mean(car_counts),
        'median_count': statistics.median(car_counts),
        'standard_deviation': statistics.stdev(car_counts),
    }

    print('Summary statistics on throughput')
    print(overall_stats)

    with open(stats_path, 'w') as csvfile:
        fieldnames = overall_stats.keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(overall_stats)

    return overall_stats


def track_throughput(boxes, predictions_save_path):
    subprocess.call(['mkdir', '-p', throughput_dir])

    # track the specific image stats
    base = os.path.splitext(os.path.basename(predictions_save_path))[0]
    unix_time = time.time()
    logs_path = os.path.join(throughput_dir, f'logs_{base}_{round(unix_time)}.csv')

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
