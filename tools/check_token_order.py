"""Check if metric PKL token ordering matches dataset ordering."""
import pickle

# Load legacy PKL (what dataset reads and sorts by timestamp)
with open(r'C:\datasets\nuscenes\nuscenes_infos_temporal_val.pkl', 'rb') as f:
    legacy = pickle.load(f)

infos = legacy['infos']
sorted_infos = sorted(infos, key=lambda e: e['timestamp'])

# Check if legacy is already sorted
already_sorted = all(
    infos[i]['timestamp'] <= infos[i+1]['timestamp']
    for i in range(len(infos)-1)
)
print(f'Legacy PKL already sorted by timestamp: {already_sorted}')

# Show first 5 tokens in original vs sorted order
print('\nOriginal order (first 5 tokens):')
for i in range(5):
    print(f'  [{i}] token={infos[i]["token"][:16]}... ts={infos[i]["timestamp"]}')

print('\nSorted order (first 5 tokens):')
for i in range(5):
    print(f'  [{i}] token={sorted_infos[i]["token"][:16]}... ts={sorted_infos[i]["timestamp"]}')

# Load metric PKL
with open(r'C:\datasets\nuscenes\nuscenes_infos_temporal_val_metric.pkl', 'rb') as f:
    metric = pickle.load(f)

metric_list = metric['data_list']

# Compare: does metric[i]['token'] == sorted_infos[i]['token'] for all i?
match_count = sum(
    1 for i in range(len(metric_list))
    if metric_list[i]['token'] == sorted_infos[i]['token']
)
mismatch_count = len(metric_list) - match_count

print(f'\nMetric PKL vs sorted dataset order:')
print(f'  Matching tokens: {match_count}/{len(metric_list)}')
print(f'  Mismatched tokens: {mismatch_count}')

if mismatch_count > 0:
    print('\n  THIS IS THE ROOT CAUSE OF mAP=0!')
    print('  Predictions for sample i are evaluated against wrong ground truth.')
    for i in range(min(5, len(metric_list))):
        if metric_list[i]['token'] != sorted_infos[i]['token']:
            print(f'  Mismatch at index {i}:')
            print(f'    Metric: {metric_list[i]["token"][:20]}')
            print(f'    Dataset: {sorted_infos[i]["token"][:20]}')
else:
    print('\n  Ordering matches - this is NOT the issue.')
