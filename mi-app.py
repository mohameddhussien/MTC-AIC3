import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

base_path = './'
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
validation_df = pd.read_csv(os.path.join(base_path, 'validation.csv'))
test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))

# Function to load a trial's EEG data
def load_trial_data(row, base_path='.'):
    # Determine dataset type based on ID range
    id_num = row['id']
    if id_num <= 4800:
        dataset = 'train'
    elif id_num <= 4900:
        dataset = 'validation'
    else:
        dataset = 'test'

    # Construct the path to EEGdata.csv
    eeg_path = f"{base_path}/{row['task']}/{dataset}/{row['subject_id']}/{row['trial_session']}/EEGdata.csv"

    # Load the entire EEG file
    eeg_data = pd.read_csv(eeg_path)

    # Calculate indices for the specific trial
    trial_num = int(row['trial'])
    if row['task'] == 'MI':
        samples_per_trial = 2250  # 9 seconds * 250 Hz
    else:  # SSVEP
        samples_per_trial = 1750  # 7 seconds * 250 Hz

    start_idx = (trial_num - 1) * samples_per_trial
    end_idx = start_idx + samples_per_trial - 1

    # Extract the trial data
    trial_data = eeg_data.iloc[start_idx:end_idx+1]
    return trial_data

# Example: Load data for the first training example
first_trial = train_df.iloc[0]  # Using index 0 for the first row
trial_data = load_trial_data(first_trial, base_path=base_path)  # Pass the base_path parameter
# print(f"""
# Loaded task: {first_trial['task']},
# Trial: {first_trial['trial']}
# for subject {first_trial['subject_id']}
# """)
print(first_trial)
# Print label if available (only for training and validation data)
if 'label' in first_trial:
    print(f"Label: {first_trial['label']}")
print(f"Data shape: {trial_data.shape}")
print(f"First 5 rows of trial data:\n{trial_data.head()}")
