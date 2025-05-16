import torch
from torch.utils.data import DataLoader, TensorDataset
import mne
from mne.preprocessing import ICA
from scipy.signal import istft
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
import hickle
import random
import tensorflow as tf

def EEG_generator (batch_size):

    train_X_train = np.load (".npy")[:]
    train_y_train = np.load (".npy")[:]

    test_X_train = np.load(".npy")[:]
    test_y_train = np.load(".npy")[:]

    print("Number of 1s:", np.count_nonzero(test_y_train == 1))
    print("Number of 0s:", np.count_nonzero(test_y_train == 0))

    test_y_train = test_y_train.astype(np.int64)

    print (test_X_train.shape)
    print (test_y_train.shape)

    n_classes = 2
    input_channels = 19

    return train_X_train, train_y_train, test_X_train, test_y_train, input_channels, n_classes

def EEG_generator_Time (batch_size):

        ICA = True

        file_name_x = "."
        dev_path = "."

        train_y_path = ".npy"

        dev_path_x = ".npy" #NO ICA
        dev_path_y = ".npy"

        if ICA:
            train_X_train = ICA_Data(file_name_x)
            test_X_train = ICA_Data(dev_path)
            test_X_train = test_X_train.astype(np.float16)

        train_X_train = train_X_train[:]  # Good
        train_y_train = np.load(train_y_path)[:]  # Good

        train_y_train = train_y_train.astype(np.int64)

        print(train_X_train.shape, train_y_train.shape)

        print("Number of 1s:", np.count_nonzero(train_y_train == 1))
        print("Number of 0s:", np.count_nonzero(train_y_train == 0))

        def Noise_Min_Max(X_test):

            new_or = X_test.reshape(-1, X_test.shape[-1])
            scaler = MinMaxScaler()
            new_or = scaler.fit_transform(new_or)

            sfreq = 250  # Sample frequency in Hz, adjust as needed
            chs = [u'FP1-F7', u'F7-T7', u'T7-P7', u'P7-O1', u'FP1-F3', u'F3-C3', u'C3-P3', u'P3-O1', u'FP2-F4',
                   u'F4-C4', u'C4-P4', u'P4-O2', u'FP2-F8', u'F8-T8', u'T8-P8', u'P8-O2', u'FZ-CZ', u'CZ-PZ',
                   u'P7-T7']

            info1 = mne.create_info(chs, sfreq, ch_types='eeg')
            raw1 = mne.io.RawArray(new_or.transpose(1, 0), info1)
            raw1.notch_filter(60, fir_design='firwin')

            filtered_data = raw1.get_data().transpose(1, 0)
            filtered_data = filtered_data.reshape(X_test.shape[0], 3000, X_test.shape[2])

            return filtered_data

        train_X_train = Noise_Min_Max(train_X_train)
        train_X_train = train_X_train.astype(np.float16)
        test_X_train = Noise_Min_Max(test_X_train[0:20000])
        test_X_train = test_X_train.astype(np.float16)

        # test_X_train = test_X_train[0:20000] #good
        test_y_train = np.load(dev_path_y)[0:20000] #good
        test_y_train = test_y_train.astype(np.int64)

        print("Number of 1s:", np.count_nonzero(test_y_train == 1))
        print("Number of 0s:", np.count_nonzero(test_y_train == 0))

        train_dataset = TensorDataset(torch.FloatTensor(train_X_train), torch.tensor(train_y_train))
        test_dataset = TensorDataset(torch.FloatTensor(test_X_train), torch.tensor(test_y_train))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # I m gonna use 120.
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # i m gonna use 120

        n_classes = 2
        seq_length = 12 * 250
        input_channels = 19

        return train_loader, test_loader, seq_length, input_channels, n_classes

def create_mne_raw(data, sfreq, chs=None):

    ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1',
           'O2']

    if chs is None:
        chs_ = ['ch{}'.format(i) for i in range(data.shape[0])]
    else:
        # assert data.shape[0] == len(chs)
        chs_ = ch_names

    ch_types = ['eeg' for _ in range(len(chs_))]

    info = mne.create_info(ch_names=chs_, sfreq=sfreq, ch_types=ch_types, verbose=False)
    print (info)
    raw = mne.io.RawArray(data * 1e-7, info)
    print ("here1")

    return raw

def ica_arti_remove(data, sfreq, chs=None):

    raw = create_mne_raw(data, sfreq, chs)
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=0.1, h_freq=None, verbose=False)

    ica = ICA(n_components=19, random_state=13)
    try:
        ica.fit(filt_raw, verbose=False)
    except:
        return None

    print ("here2")

    ica.exclude = []

    eog_indices1, eog_scores1 = ica.find_bads_eog(filt_raw, threshold=2, ch_name='Fp1', verbose=False)
    print('eog_indices1', eog_indices1)
    eog_indices2, eog_scores2 = ica.find_bads_eog(filt_raw, threshold=2, ch_name='Fp2', verbose=False)
    print('eog_indices2', eog_indices2)

    if len(eog_indices1) > 0:
        ica.exclude.append(eog_indices1[0])
    if len(eog_indices2) > 0:
        ica.exclude.append(eog_indices2[0])

    print('ica.exclude', ica.exclude)

    if len(ica.exclude) > 0:
        reconst_raw = filt_raw.copy()
        reconst_raw.load_data()
        ica.apply(reconst_raw)
        print('Reconstructing data from ICA components...')
        return reconst_raw.get_data() * 1e6

    return data

def Segmentation (data_processed):

    segment_duration_samples = 12 * 250
    num_segments = int (data_processed.shape[1]//segment_duration_samples)
    segmented_data_shape = (19, 3000, num_segments)
    segmented_data = np.zeros(segmented_data_shape)

    for i in range(num_segments):

        start_idx = i * segment_duration_samples
        end_idx = (i + 1) * segment_duration_samples
        segment = data_processed[:, start_idx:end_idx]
        segmented_data[:,:,i] = segment

    segmented_data = segmented_data.transpose(2, 1, 0)

    return segmented_data

def initialize_savings (data, output_fol):

    start_idx1= 1000
    print (data.shape[0])
    n = int(data.shape[0]/start_idx1)
    for i in range(n):
        start_idx = i * start_idx1 #0,
        end_idx = (i + 1) * start_idx1 #1000
        data1 = data[start_idx:end_idx]
        data2 = data1.transpose(2, 0, 1).reshape(19, -1) #19, 3000 x length.
        data_processed = ica_arti_remove(data2, 250, chs=19)
        segmented_data = Segmentation(data_processed)
        segmented_data = segmented_data.astype(np.float16)
        np.save (output_fol + str(i) +"_subset.npy", segmented_data)

def extract_number(filename):
    """Extracts the number from the filename."""
    match = re.search(r'(\d+)_subset', filename)
    return int(match.group(1)) if match else None

def ICA_Data(directory):

    files = [f for f in os.listdir(directory) if f.endswith('.npy')]
    files.sort(key=extract_number)

    # List to store the loaded arrays
    arrays = []

    # Loop through each sorted file and load the array
    for filename in files:
        file_path = os.path.join(directory, filename)
        print (filename)
        data = np.load(file_path)
        arrays.append(data)

    # Stack the arrays along the first dimension
    final_array = np.vstack(arrays)

    # Print the final shape
    print("Final shape:", final_array.shape)
    return final_array

def Epilepsia_12s_STFT(patname,batch_size):


    test_X_train =  np.load("/mnt/data13_16T/EPILEPSIAE_SCALP/"+ patname+ "_TestX.npy")
    print (test_X_train.shape)
    test_X_train = np.transpose(test_X_train, (0,3,1,2)) #EPILEPSIAE
    print (test_X_train.shape)

    test_X_train = float16_to_uint8(test_X_train) ##this is for the quantization part.

    test_y_train = np.load("/mnt/data13_16T/EPILEPSIAE_SCALP/" + patname + "_TestY.npy")

    print("Number of 1s:", np.count_nonzero(test_y_train == 1))
    print("Number of 0s:", np.count_nonzero(test_y_train == 0))

    test_dataset = TensorDataset(torch.FloatTensor(test_X_train), torch.tensor(test_y_train))

    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                         shuffle=True)

    return test_loader, test_X_train, test_y_train


    test_X_train = np.load("/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/"+year+"/"+patname+"/totalx.npy")[:]
    test_y_train = np.load("/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/"+year+"/"+patname+"/totaly.npy")[:]

    test_X_train = test_X_train.astype(np.float16)

    # print("Number of 1s:", np.count_nonzero(test_y_train == 1))
    # print("Number of 0s:", np.count_nonzero(test_y_train == 0))

    test_y_train = test_y_train.astype(np.int64)

    test_X_train = np.transpose(test_X_train, (0, 2, 1, 3))

    print (test_X_train.shape)
    print (test_y_train.shape)

    # label_indices_to_delete = np.where(test_y_train == 1)[0]
    #
    # if len(label_indices_to_delete) > 20 and (year == "2014" or year == "2015"):
    #      indices_to_keep = np.random.choice(label_indices_to_delete, size=20, replace=False)
    #     indices_to_delete = np.setdiff1d(label_indices_to_delete, indices_to_keep)
    #      test_X_train = np.delete(test_X_train, indices_to_delete, axis=0)
    #     test_y_train = np.delete(test_y_train, indices_to_delete)

    print("Number of 1s:", np.count_nonzero(test_y_train == 1))
    print("Number of 0s:", np.count_nonzero(test_y_train == 0))

    test_dataset = TensorDataset(torch.FloatTensor(test_X_train), torch.tensor(test_y_train))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return test_loader


def float16_to_uint8(image):
    """
    Convert a float16 image to uint8 by linear normalization.

    Parameters:
        image (np.ndarray): Input image of dtype float16.

    Returns:
        np.ndarray: Normalized image of dtype uint8.
    """
    # Ensure the image is in float16
    # image = image.astype(np.float16)

    # Find the min and max values
    img_min = image.min()
    img_max = image.max()

    # Avoid division by zero in case the image is constant
    if img_max == img_min:
        return np.zeros_like(image, dtype=np.uint8)

    # Normalize to [0, 1]
    normalized = (image - img_min) / (img_max - img_min)

    # Scale to [0, 255] and convert to uint8
    scaled = (normalized * 15.0).round().astype(np.uint8)

    return scaled

def RPA_generator_KERAS(shots, patname, year, batch_size):

    tf.random.set_seed(42)  # for reproducibility.
    random.seed(42)  # for reproducibility.

    data_path = f"/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/{year}/pat{patname}"
    X = np.load(f"{data_path}/totalx.npy")[:]  # e.g., shape: (samples, ?, ?, ?)
    y = np.load(f"{data_path}/totaly.npy")[:]  # e.g., shape: (samples,)

    # Convert data types.
    X = X.astype(np.float16)
    X = float16_to_uint8(X)
    # X = np.clip(X, 0, 255).astype(np.uint8)
    y = y.astype(np.int64)

    # Rearrange dimensions if needed.
    X = np.transpose(X, (0, 2, 3, 1))
    print("Data shape:", X.shape) #(Batch, 19 channels, 123 fr, 12 time bins)

    class_ids = [0, 1]
    class_samples = {cls: [] for cls in class_ids}

    # Group samples by class.
    for i in range(len(y)):
        label = y[i]
        if label in class_samples:
            class_samples[label].append(X[i])

    train_dict = {}      # To hold the "shots" samples for training.
    inference_dict = {}  # To hold the remaining samples for inference.

    # For each class, select 'shots' for training and assign the rest for inference.
    for cls in class_ids:
        all_samples = np.array(class_samples[cls])
        available = len(all_samples)
        print(f"Class {cls} has {available} samples available.")

        if available == 0:
            print(f"Warning: No samples available for class {cls}.")
            train_dict[cls] = np.empty((0,) + all_samples.shape[1:])
            inference_dict[cls] = np.empty((0,) + all_samples.shape[1:])
            continue

        if available < shots:
            print(f"Warning: Class {cls} has only {available} sample(s). Requested {shots} shot(s). Using all available samples for training.")
            train_dict[cls] = all_samples
            inference_dict[cls] = np.empty((0,) + all_samples.shape[1:])  # No remaining samples.
        else:
            indices = list(range(available))
            train_indices = random.sample(indices, shots)
            inference_indices = [i for i in indices if i not in train_indices]

            train_dict[cls] = all_samples[train_indices]
            inference_dict[cls] = all_samples[inference_indices]

    # Return a dictionary containing training and inference data.
    return train_dict, inference_dict


def split_dataset_into_shot_and_inference(patname, year, shot_count):

    np.random.seed(42)
    # tf.random.set_seed(42)  # for reproducibility.

    # data_path = f"/mnt/data7_4T/temp/yikai/RPA_AUC_stft_ICA_totalPat/{year}/pat{patname}"
    ###EPILEPSIAE
    data_path = f"/mnt/data13_16T/EPILEPSIAE_SCALP/{patname}"

    X = np.load(f"{data_path}_TestX.npy")[:]
    y = np.load(f"{data_path}_TestY.npy")[:]

    # Convert data types.
    X = X.astype(np.float16)
    X = float16_to_uint8(X)
    Y = y.astype(np.int32)

    print (X.shape)
    print (Y.shape)

    # Rearrange dimensions if needed.  #ONLY FOR RPA
    # X = np.transpose(X, (0, 2, 3, 1))
    X = np.transpose(X, (0,3,1,2)) #EPILEPSIAE
    print("Data shape:", X.shape)  # (Batch, 19 channels, 123 fr, 12 time bins)

    shot_data = {0: [], 1: []}  # Explicit keys for seizure (1) and non-seizure (0)
    inference_data = {0: [], 1: []}

    # Iterate over each class label (0: Non Seizure, 1: Seizure)
    for label in np.unique(Y):
        class_indices = np.where(Y == label)[0] #find all the indices
        num_samples = len(class_indices)   # print(f"Class {label}: Found {num_samples} samples.")

        if num_samples == 0:
            print(f"Warning: No samples found for class {label}. Skipping.")
            continue

        actual_shot_count = min(shot_count, num_samples)  # Ensure shot_count does not exceed available samples

        if actual_shot_count < shot_count:
            print(f"Warning: Only {actual_shot_count} samples available for class {label}.")

        # Randomly select shot samples
        if actual_shot_count > 0:
            shot_indices = np.random.choice(class_indices, size=actual_shot_count, replace=False)
        else:
            shot_indices = []

        inf_indices = np.setdiff1d(class_indices, shot_indices)

        # Ensure indexing works
        if len(shot_indices) > 0:
            print(f"Class {label}: Shot sample shape from X: {X[shot_indices].shape}")

        # Assign data correctly
        shot_data[label] = list(X[shot_indices]) if len(shot_indices) > 0 else []
        inference_data[label] = list(X[inf_indices])

    print(f"Final Shot Samples Per Class: { {k: len(v) for k, v in shot_data.items()} }")
    print(f"Final Inference Samples Per Class: { {k: len(v) for k, v in inference_data.items()} }")

    # shot_data = {
    #     'X': X[shot_indices],
    #     'Y': Y[shot_indices]
    # }
    # inference_data = {
    #     'X': X[inference_indices],
    #     'Y': Y[inference_indices]
    # }
    return shot_data, inference_data




