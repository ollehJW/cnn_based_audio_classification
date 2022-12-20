## 1. Import Packages
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from data_utils import construct_filesets, make_npy_files, make_fileset_indexing
from data_utils.data_generator import ConstructDataset
from models import ModelFactory, SupervisedTraining

## 2. Transform wav to npy.
### (1) Parameter Setting
SR = 16000
IMG_HEIGHT = 224
IMG_WIDTH = 224
audio_file_path = './dataset/audio_files'

### (2) Construct Filesets
train_filesets, train_uni_label, train_labels = construct_filesets(os.path.join(audio_file_path, 'train'))
test_filesets, test_uni_label, test_labels = construct_filesets(os.path.join(audio_file_path, 'test'))

### (3) Make npy save path
print("Make npy Folders")
npy_file_path = audio_file_path.replace('audio_files', 'npy_files')
os.makedirs(npy_file_path, exist_ok=True)
os.makedirs(os.path.join(npy_file_path, 'train'), exist_ok=True)
os.makedirs(os.path.join(npy_file_path, 'test'), exist_ok=True)

for ul in train_uni_label:
    os.makedirs(os.path.join(os.path.join(npy_file_path, 'train'), ul), exist_ok=True)

for ul in test_uni_label:
    os.makedirs(os.path.join(os.path.join(npy_file_path, 'test'), ul), exist_ok=True)

### (4) Make npy files
if os.path.getsize(npy_file_path) > 0:
    print("npy Files already exists.")
else:
    print("Convert Audio files to Spectrogram npy files... (Train filesets)")
    for audio_path in tqdm(train_filesets):
        make_npy_files(audio_path, sr = SR, img_height = IMG_HEIGHT, img_width = IMG_WIDTH)

    print("Convert Audio files to Spectrogram npy files... (Test filesets)")
    for audio_path in tqdm(test_filesets):
        make_npy_files(audio_path, sr = SR, img_height = IMG_HEIGHT, img_width = IMG_WIDTH)

train_filesets = [filesets.replace('audio_files', 'npy_files') for filesets in train_filesets]
train_filesets = [filesets.replace('.wav', '.npy') for filesets in train_filesets]
test_filesets = [filesets.replace('audio_files', 'npy_files') for filesets in test_filesets]
test_filesets = [filesets.replace('.wav', '.npy') for filesets in test_filesets]

    

## 3. Data Generator
### (1) Parameter Setting
AUG_MIXUP = True
AUG_ROLLING = True
AUG_SPEC = True

BATCH_SIZE = 32

### (2) Construct Torch Dataset 
print("Construct Torch Dataset")
train_dataset = ConstructDataset(train_filesets, train_uni_label, train_labels, AUG_MIXUP, AUG_ROLLING, AUG_SPEC)
test_dataset = ConstructDataset(test_filesets, test_uni_label, test_labels, False, False, False)

### (3) Data Loader
print("Construct Torch Dataloader")
train_dataloader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
train_dataloader.dataset.file_indexing(seed = 1004)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
test_dataloader.dataset.file_indexing(seed = 1004)

## 4. Classificaiton Model
### (1) Parameter Setting
MODEL_ARCHITECTURE = 'mobilenet_v2'
EPOCH = 20
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-6
MODEL_PATH = './result'
os.makedirs(MODEL_PATH, exist_ok=True)

### (2) Construct Model
print("Construct {} Model.".format(MODEL_ARCHITECTURE))
model = ModelFactory(model_name=MODEL_ARCHITECTURE, class_num=len(train_uni_label))
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr = LEARNING_RATE, weight_decay=WEIGHT_DECAY)

### (3) Train
print("Start Training!!")
trainer = SupervisedTraining(epoch=EPOCH, result_model_path = MODEL_PATH)
trainer.train(model, train_dataloader, test_dataloader, criterion, optimizer, gpu = True)

