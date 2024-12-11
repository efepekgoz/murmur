import os
import pandas as pd
import numpy as np
import wave
from pydub import AudioSegment
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from os.path import exists
from tqdm import tqdm
from torchvision import models, transforms

# Check path
data_csv = r'/Users/efepekgoz/Developer/Murmur_3/training_data.csv'

df = pd.read_csv(data_csv)
print(f"Initial dataset size: {df.shape}")

columns_to_drop = [
    'Age', 'Sex', 'Height', 'Weight', 'Pregnancy status', 'Most audible location', 
    'Systolic murmur pitch', 'Systolic murmur quality', 'Diastolic murmur timing', 
    'Diastolic murmur shape', 'Diastolic murmur grading', 'Diastolic murmur pitch', 
    'Diastolic murmur quality', 'Outcome', 'Campaign', 'Additional ID', 
    'Systolic murmur grading', 'Systolic murmur shape', 'Systolic murmur timing'
]

df.drop(columns=columns_to_drop, inplace=True)

# Printing dataset size after dropping columns
''' Columns will drop but the rows will stay so expect a lower dimension in terms of columns'''
print(f"Dataset size after dropping columns: {df.shape}")

# Prepare the dataset based on multiple recording locations per patient
''' patients have MV+PV+TV recordings as listed in excel sheet'''
rows = []
for index, row in df.iterrows():
    try:
        recording_locations = row['Recording locations:'].split('+')
        patient_id = row['Patient ID']
        murmur = row['Murmur']
        murmur_locations = row['Murmur locations'].split('+') if isinstance(row['Murmur locations'], str) else []

# Check filepath for recordings
        for recording_loc in recording_locations:
            filepath = fr'/Users/efepekgoz/Developer/Murmur_3/training_data/{patient_id}_{recording_loc}.wav'
            location_map = {"AV": 1, "PV": 2, "TV": 3, "MV": 4}
            # Default to 5 for other locations ; Misc category
            integer_add = location_map.get(recording_loc, 5)  
            label = 1 if recording_loc in murmur_locations else 0

            rows.append({'Patient ID': patient_id, 'Label': label, 'Recording locations': integer_add, 'filepath': filepath})
    except KeyError as e:
        print(f"KeyError: 'Recording locations' not found in row {index}: {e}")

new_df = pd.DataFrame(rows)
print(f"Processed DataFrame size: {new_df.shape}")

# Verify and process audio files, handling missing files and processing valid ones
missing_files = []
processed_files = []

for index, row in new_df.iterrows():
    filepath = row['filepath']
    
    # Verify the existence of the WAV file
    if not os.path.exists(filepath):
        missing_files.append(filepath)
        print(f"File not found: {filepath}")
        continue  # Skip processing if the file does not exist
    
    # Prepare the path for the corresponding TSV annotation file
    tsv_filepath = filepath.replace('.wav', '.tsv')
    
    # Check the existence of the TSV file
    if not os.path.exists(tsv_filepath):
        print(f"TSV file not found: {tsv_filepath}")
        missing_files.append(tsv_filepath)
        continue  # Skip processing if the TSV file does not exist

    # Attempt to load and process the TSV and corresponding audio file
    try:
        tsv_data = pd.read_csv(tsv_filepath, sep='\t', names=['start', 'end', 'label'])
        audio = AudioSegment.from_file(filepath)
        segments = []

        # Extract segments specified in the TSV and concatenate them
        for _, seg in tsv_data.iterrows():
            start_ms = seg['start'] * 1000  # Convert start time to milliseconds
            end_ms = seg['end'] * 1000      # Convert end time to milliseconds
            segments.append(audio[start_ms:end_ms])

        combined = segments[0]
        for segment in segments[1:]:
            combined += segment

        # Save the combined audio back to the original file
        combined.export(filepath, format='wav')
        processed_files.append(filepath)
        print(f"Processed and saved spliced audio: {filepath}")

    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")

# It will print all the missing files found
print(f"Total missing files: {len(missing_files)}")
if missing_files:
    print("Examples of missing files:", missing_files[:10])
print(f"Total processed files: {len(processed_files)}")

# Initialize a list for indicies that need to be dropped 
indices_to_drop = []

# Iterate over each row in the DataFrame, checking the status of audio files.
for index, row in new_df.iterrows():
    filepath = row['filepath']
    
    # Check if the audio file exists at the specified path.
    if not exists(filepath):
        # If the file does not exist, add the index of this row to the list of indices to drop.
        indices_to_drop.append(index)
        print(f"File does not exist, dropping index: {index}")
        continue  
    
    
    try:
        wav_file = AudioSegment.from_file(filepath)
        
        # Check if the audio file has zero length, indicating an empty or invalid file.
        if len(wav_file) == 0:
            # If the file is zero length, add the index to the list to be dropped.
            indices_to_drop.append(index)
            print(f"Zero length audio at index {index}, dropping.")
    except Exception as e:
        # Catch any exceptions raised during audio file processing and mark the index for dropping.
        indices_to_drop.append(index)
        print(f"Failed to load audio at index {index}, error: {e}")

# Drop the rows from the DataFrame at the collected indices due to missing or invalid audio files.
new_df = new_df.drop(indices_to_drop).reset_index(drop=True)

# Print the size of the DataFrame after dropping indicies.
print(f"Dataset size after removing zero-length audio files: {new_df.shape}")

#train test val split
train_size = 0.7  
validation_size = 0.15  
test_size = 0.15 


train, temp = train_test_split(new_df, train_size=train_size, random_state=42)

valid, test = train_test_split(temp, test_size=test_size / (validation_size + test_size), random_state=42)

print("DataFrame shape before split:", new_df.shape)
print("Training set size:", train.shape)
print("Validation set size:", valid.shape)
print("Testing set size:", test.shape)

valid.head()

# Get longest audio file
"""
longest_duration = 0
longest_path = ''
for index, row in new_df.iterrows():
  file_path = row['filepath']
  try:
    wav,sr = librosa.load(file_path,sr=None)
    duration = librosa.get_duration(y=wav, sr=sr)
  except:
    # Combine two files
    print("combine")
    combine_sound(file_path)
    wav,sr = librosa.load(file_path,sr=None)
    duration = librosa.get_duration(y=wav, sr=sr)
  if duration > longest_duration:
    longest_duration = duration
    longest_path = file_path
print(longest_duration, longest_path)
"""

device = torch.device("cuda" if torch.cuda.is_available() else "mps") #or cpu
print(device)

import cv2

def spec_to_image(spec, eps=1e-6):
  mean = spec.mean()
  std = spec.std()
  spec_norm = (spec - mean) / (std + eps)
  spec_min, spec_max = spec_norm.min(), spec_norm.max()
  spec_scaled = 255 * (spec_norm - spec_min) / (spec_max - spec_min)
  spec_scaled = spec_scaled.astype(np.uint8)
# inception resize
  spec_resized = cv2.resize(spec_scaled, dsize=(299,299),interpolation=cv2.INTER_CUBIC)
  spec_scaled = spec_resized
  return spec_scaled

def combine_sound(file_path):
  infiles = [file_path.split(".wav")[0] + "_1.wav", file_path.split(".wav")[0] + "_2.wav"]

  outfile = file_path

  data= []
  for infile in infiles:
      w = wave.open(infile, 'rb')
      data.append( [w.getparams(), w.readframes(w.getnframes())] )
      w.close()
      
  output = wave.open(outfile, 'wb')
  output.setparams(data[0][0])
  for i in range(len(data)):
      output.writeframes(data[i][1])
  output.close()

## Mel Spectogram

def get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80):
    wav, sr = librosa.load(file_path, sr=sr)
    # Pad or truncate wav file to have a consistent length (18 seconds in this example)
    if wav.shape[0] < 18 * sr:
        wav = np.pad(wav, (0, max(0, 18 * sr - wav.shape[0])), mode='reflect')
    else:
            wav = wav[:18 * sr]
    spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, ref=np.max, top_db=top_db)
    return spec_db

## Mel Spectogram with transformation

def get_melspectrogramTransforms_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80, pitch_shift_steps=0, speed_change_factor=1.0, noise_level=0.005):
    wav, sr = librosa.load(file_path, sr=sr)
    # Pitch shifting
    if pitch_shift_steps != 0:
        wav = librosa.effects.pitch_shift(wav, sr=sr, n_steps=pitch_shift_steps)
    # Speed change
    if speed_change_factor != 1.0:
        wav = librosa.effects.time_stretch(wav, rate=speed_change_factor)
    # Add noise
    wav += noise_level * np.random.randn(len(wav))

    # Pad or truncate wav file to have a consistent length (18 seconds in this example)
    if wav.shape[0] < 18 * sr:
        wav = np.pad(wav, (0, max(0, 18 * sr - wav.shape[0])), mode='reflect')
    else:
        wav = wav[:18 * sr]
    spec = librosa.feature.melspectrogram(y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=fmin, fmax=fmax)
    spec_db = librosa.power_to_db(spec, ref=np.max, top_db=top_db)
    return spec_db

# STFT Spectogram

def get_stft_spectrogram(file_path, sr=None, hop_length=1024, n_fft=2048, target_length_sec=18):
    wav, sr = librosa.load(file_path, sr=sr)
    # Ensure the audio is exactly 18 seconds long
    target_length = sr * target_length_sec
    if len(wav) < target_length:
        wav = np.pad(wav, (0, target_length - len(wav)), 'constant')
    else:
        wav = wav[:target_length]
    # Compute the STFT
    D = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
    # Convert the magnitude spectrogram (D) to decibel (dB) units
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    return D_db


## LOG Spectrogram

def get_log_spectrogram(file_path, sr=None, hop_length=1024, n_fft=2048, target_length_sec=18):
    wav, sr = librosa.load(file_path, sr=sr)
    # Ensure the audio is exactly 18 seconds long
    target_length = sr * target_length_sec
    if len(wav) < target_length:
        wav = np.pad(wav, (0, target_length - len(wav)), 'constant')
    else:
        wav = wav[:target_length]
    D = np.abs(librosa.stft(wav, n_fft=n_fft, hop_length=hop_length))
    D_db = librosa.amplitude_to_db(D, ref=np.max)
    return D_db


class HeartMurmurData(Dataset):
    def __init__(self, df, in_col, out_col, spec_type='mel', pitch_shift_steps=0, speed_change_factor=1.0, noise_level=0.005):
        self.df = df
        self.data = []
        self.labels = []
        self.c2i = {}
        self.i2c = {}
        self.spec_type = spec_type
        self.pitch_shift_steps = pitch_shift_steps
        self.speed_change_factor = speed_change_factor
        self.noise_level = noise_level

        self.categories = sorted(df[out_col].unique())
        for i, category in enumerate(self.categories):
            self.c2i[category] = i
            self.i2c[i] = category
        
        for ind in tqdm(range(len(df))):
            row = df.iloc[ind]
            file_path = row[in_col]

            # Now include the pitch_shift_steps, speed_change_factor, and noise_level in the spectrogram function call
            if spec_type == 'melTransform':
                spectrogram = get_melspectrogramTransforms_db(file_path, pitch_shift_steps=self.pitch_shift_steps, speed_change_factor=self.speed_change_factor, noise_level=self.noise_level)
            if spec_type == 'mel':
                spectrogram = get_melspectrogram_db(file_path, sr=None, n_fft=2048, hop_length=512, n_mels=128, fmin=20, fmax=8300, top_db=80)
            elif spec_type == 'stft':
                spectrogram = get_stft_spectrogram(file_path, sr=None, hop_length=1024, n_fft=2048, target_length_sec=18)
            elif spec_type == 'log':
                spectrogram = get_log_spectrogram(file_path, sr=None, hop_length=1024, n_fft=2048, target_length_sec=18)

            self.data.append(spec_to_image(spectrogram)[np.newaxis, ...])
            self.labels.append(self.c2i[row[out_col]])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

# With Transformation MelTransform Run the below: !!
# train_data = HeartMurmurData(train, 'filepath', 'Label', pitch_shift_steps=4.0, speed_change_factor=1.1, noise_level=0.002)

# Without transform run the below line and comment the above: !!
train_data = HeartMurmurData(train, 'filepath', 'Label')

valid_data = HeartMurmurData(valid, 'filepath', 'Label')

# Determining GPU avaliability ; Need GPU 
device = torch.device('cuda:0' if torch.cuda.is_available() else 'mps')

from torchvision.models.inception import Inception3

class ModifiedInception3(Inception3):
    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False, init_weights=True):
        super(ModifiedInception3, self).__init__(
            num_classes=num_classes, aux_logits=aux_logits,
            transform_input=transform_input, init_weights=init_weights
        )
        #aux classifier disabled
        self.aux_logits = False

    def _transform_input(self, x):
        if self.transform_input:
            #making grayscale into rgb by *3 the channel. Rest is same with the original 
            x = x.repeat(1, 3, 1, 1)
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def forward(self, x):
        x = self._transform_input(x)
        x = super(ModifiedInception3, self).forward(x)
        # Return only the main output regardless of aux_logits configuration
        if isinstance(x, tuple):
            return x[0]  # Return the main output only
        return x

incep_model = ModifiedInception3()

#pretrained=True but fancier
pretrained_model = models.inception_v3(pretrained=True)
incep_model.load_state_dict(pretrained_model.state_dict(), strict=False)

#first layer and final layer modificaiton
incep_model.Conv2d_1a_3x3 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=3, bias=False)
incep_model.fc = nn.Linear(2048,2)
incep_model = incep_model.to(device)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=32, shuffle=True)
torch.cuda.empty_cache()


learning_rate = 3e-4
optimizer = optim.Adam(incep_model.parameters(), lr=learning_rate)
epochs = 20
loss_fn = nn.CrossEntropyLoss().to(device)
incep_train_losses=[]
incep_valid_losses=[]


def lr_decay(optimizer, epoch):
  if epoch%10==0:
    new_lr = learning_rate / (10**(epoch//10))
    optimizer.param_groups[0]['lr'] = new_lr
    #optimizer = setlr(optimizer, new_lr)
    print(f'Changed learning rate to {new_lr}')
  return optimizer

def train(model, loss_fn, train_loader, valid_loader, epochs, optimizer, train_losses, valid_losses, change_lr=None):
    for epoch in tqdm(range(1, epochs+1)):
        model.train()
        train_loss, valid_loss = [], []
        train_correct, train_total, valid_correct, valid_total = 0, 0, 0, 0

        # Training phase
        for data, target in train_loader:
            #print(data.shape)
            data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.long)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            _, preds = torch.max(output, 1)
            train_correct += torch.sum(preds == target).item()
            train_total += target.size(0)

        # Validation phase
        model.eval()
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.long)
                output = model(data)
                loss = loss_fn(output, target)
                
                valid_loss.append(loss.item())
                _, preds = torch.max(output, 1)
                valid_correct += torch.sum(preds == target).item()
                valid_total += target.size(0)

        # Calculate average losses and accuracy
        train_loss_avg = np.mean(train_loss)
        valid_loss_avg = np.mean(valid_loss)
        train_acc = train_correct / train_total
        valid_acc = valid_correct / valid_total
        
        train_losses.append(train_loss_avg)
        valid_losses.append(valid_loss_avg)

        # Adjust learning rate if applicable
        if change_lr:
            optimizer = change_lr(optimizer, epoch)
        
        # Print epoch summary
        print(f'Epoch {epoch:3} | Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | Valid Loss: {valid_loss_avg:.4f} | Valid Acc: {valid_acc:.4f} | LR: {optimizer.param_groups[0]["lr"]}')

train(incep_model, loss_fn, train_loader, valid_loader, epochs, optimizer, incep_train_losses, incep_valid_losses, lr_decay)

test_data = HeartMurmurData(test, 'filepath', 'Label')
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

# Checking the number of samples in the test DataLoader
test_samples_count = len(test_data)
print(f"Total samples in test dataset: {test_samples_count}")

# Calculating the number of batches and samples processed
batches_count = 0
samples_processed = 0
for data, labels in test_loader:
    batches_count += 1
    samples_processed += data.size(0)

print(f"Total batches: {batches_count}")
print(f"Total samples processed during testing: {samples_processed}")


correct = 0
predictions_ar = []
labels_ar = []

# Set the model to evaluation mode
incep_model.eval()

# Iterate over the test dataset
for data, labels in test_loader:
    # Move the data and labels to the GPU if available
    #if torch.cuda.is_available():
    data, labels = data.to(device), labels.to(device)

    # Compute the model's predictions
    outputs = incep_model(data.float())
    _, predicted = torch.max(outputs, 1)

    # Store predictions and true labels for accuracy calculation
    predictions_ar.extend(predicted.cpu().numpy())
    labels_ar.extend(labels.cpu().numpy())

# Calculate accuracy
correct = (np.array(predictions_ar) == np.array(labels_ar)).sum()
accuracy = correct / len(labels_ar)
print(f"Accuracy: {accuracy:.4f}")

# Confusion Matrix plot

conf_mat = confusion_matrix(labels_ar, predictions_ar)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap='Blues', xticklabels=["Absent", "Present"], yticklabels=["Absent", "Present"])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import f1_score

myf1 = f1_score(predictions_ar,labels_ar)
print(myf1)