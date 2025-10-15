import os
import numpy as np

FEATURES_PATH = 'features'
AUGMENTATIONS_PER_FILE = 5

def add_noise(sequence, std_dev=0.02):
    noise = np.random.normal(0, std_dev, sequence.shape)
    return sequence + noise

def random_frame_drop(sequence, min_keep=0.8):
    n_frames = len(sequence)
    keep_mask = np.random.rand(n_frames) < min_keep
    augmented_seq = sequence[keep_mask]
    while len(augmented_seq) < 5:
        augmented_seq = np.vstack([augmented_seq, np.zeros(augmented_seq.shape[1])])
    return augmented_seq

def random_frame_duplicate(sequence, max_dupes=3):
    # Randomly duplicate some frames up to max_dupes times
    output = []
    for frame in sequence:
        output.append(frame)
        if np.random.rand() < 0.2:
            dupes = np.random.randint(1, max_dupes+1)
            for _ in range(dupes):
                output.append(frame)
    return np.array(output)

def temporal_shift(sequence, max_shift=5):
    # Shift sequence to left or right by up to max_shift frames, pad with zeros
    shift = np.random.randint(-max_shift, max_shift+1)
    if shift == 0:
        return sequence
    if shift > 0:
        shifted = np.vstack([np.zeros((shift, sequence.shape[1])), sequence[:-shift]])
    else:
        shifted = np.vstack([sequence[-shift:], np.zeros((-shift, sequence.shape[1]))])
    return shifted

def random_scale(sequence, scale_range=(0.95, 1.05)):
    scale_factor = np.random.uniform(*scale_range)
    return sequence * scale_factor

for root, dirs, files in os.walk(FEATURES_PATH):
    for file in files:
        if file.endswith('.npy') and '_aug' not in file:
            original_path = os.path.join(root, file)
            sequence = np.load(original_path)

            for i in range(1, AUGMENTATIONS_PER_FILE + 1):
                seq = add_noise(sequence)
                seq = random_frame_drop(seq)
                seq = random_frame_duplicate(seq)
                seq = temporal_shift(seq)
                seq = random_scale(seq)
                
                # Ensure minimum length after augmentations, pad if necessary
                if len(seq) < 5:
                    seq = np.vstack([seq, np.zeros((5 - len(seq), seq.shape[1]))])
                
                save_name = file.replace('.npy', f'_aug{i}.npy')
                save_path = os.path.join(root, save_name)
                
                np.save(save_path, seq)
                print(f'Augmented sample saved: {save_path}')

print("Advanced data augmentation complete.")
