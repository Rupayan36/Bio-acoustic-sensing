import os
import time
import numpy as np
import pandas as pd
from pathlib import Path
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from joblib import Parallel, delayed
import multiprocessing
import json
import albumentations as A
import cv2
import torchaudio
import torch.nn as nn
import torchvision.models as models

# -----------------------------------------------------------------------------
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OMP_SCHEDULE"] = "STATIC"

inferenceStartTime = time.time()
checkTimePerBatchTime = 6000  # 100 min
checkTimePerBatch = False
maxInferenceTime = 7080       # 118 min
maxInferenceTimeReached = False

batchSizeForTesting = 8
nWorkers = multiprocessing.cpu_count()

applylPostProcessing = True
neighbouringAggregationFactor = 1.0  # tweak as needed

dataDir = '../input/birdclef-2024/test_soundscapes/'
submissionMode = len(list(Path(dataDir).glob('*.ogg'))) > 1
print('submissionMode', submissionMode)
print('dataDir', dataDir)

# Audio processing parameters
sampleRate = 32000
mel_spec_params = {
    "sample_rate": sampleRate,
    "n_mels": 64,
    "f_min": 20,
    "f_max": 16000,
    "n_fft": 2048,
    "hop_length": 1024,
    "normalized": True,
    "center": True,
    "pad_mode": "constant",
    "norm": "slaney",
    "mel_scale": "slaney"
}
top_db = 80
db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=top_db)

# -----------------------------------------------------------------------------
# spectrogram extraction
# -----------------------------------------------------------------------------
def normalize_melspec(X, eps=1e-6):
    mean = X.mean((1, 2), keepdim=True)
    std = X.std((1, 2), keepdim=True)
    return (X - mean) / (std + eps)

def getSpecImageTorch(sampleVec):
    sampleVecTensor = torch.tensor(sampleVec, dtype=torch.float32).unsqueeze(0)
    melSpecParams = mel_spec_params.copy()
    melSpecParams['n_mels'] = 64
    melSpecParams['hop_length'] = 1024
    melSpec = torchaudio.transforms.MelSpectrogram(**melSpecParams)(sampleVecTensor)
    melSpec = normalize_melspec(db_transform(melSpec)).squeeze(0).numpy()
    return melSpec


# -----------------------------------------------------------------------------
# Resizing spectrogram images to 224x224 (for ResNet input)
class AudioDatasetTorchPrePro(Dataset):
    def __init__(self, rowIxs, specImages, transform=None, imageHeight=224, imageWidth=224):
        self.rowIxs = rowIxs
        self.specImages = specImages
        self.transform = transform
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth

    def __len__(self):
        return len(self.rowIxs)

    def __getitem__(self, idx):
        rowIx = self.rowIxs[idx]
        specImage = self.specImages[rowIx]
        specImage = cv2.resize(specImage, (self.imageWidth, self.imageHeight), interpolation=cv2.INTER_LINEAR)
        specImage = (specImage * 255).astype(np.uint8)
        specImage = np.stack([specImage] * 3, axis=-1)
        if self.transform:
            res = self.transform(image=specImage)
            specImage = res['image']
        specImage = specImage.transpose(2, 0, 1)
        return {'specImage': specImage, 'rowIx': rowIx}

# -----------------------------------------------------------------------------
# prepare test dataframe and extract spectrograms
# -----------------------------------------------------------------------------
def get_test_df():
    filenames, paths, row_ids, start_times, end_times = [], [], [], [], []
    for file in os.listdir(dataDir):
        if file.endswith('.ogg'):
            path = os.path.join(dataDir, file)
            filename = os.path.splitext(file)[0]
            for end_time in range(5, 245, 5):
                row_id = f"{filename}_{end_time}"
                filenames.append(filename)
                paths.append(path)
                row_ids.append(row_id)
                start_times.append(end_time - 5)
                end_times.append(end_time)
    return pd.DataFrame({
        'filename': filenames,
        'path': paths,
        'row_id': row_ids,
        'start_time': start_times,
        'end_time': end_times
    })

def getSpecImagesPerFile(path, sampleRate=32000, segmentDuration=5.0):
    sampleVecFile, _ = sf.read(path)
    specImages = []
    durationFile = 240.0
    durationFileInSamples = int(durationFile * sampleRate)
    segmentDurationInSamples = int(segmentDuration * sampleRate)
    for startSample in range(0, durationFileInSamples, segmentDurationInSamples):
        endSample = startSample + segmentDurationInSamples
        sampleVec = sampleVecFile[startSample:endSample]
        mel_spectrogram = getSpecImageTorch(sampleVec)
        specImages.append(mel_spectrogram)
    return specImages

def getSpecImages(test_df):
    paths = test_df.path.unique()
    print('nFilesToProcess', len(paths))
    with Parallel(n_jobs=nWorkers) as parallel:
        specImagesPerFile = parallel(delayed(getSpecImagesPerFile)(path) for path in paths)
    specImages = [item for sublist in specImagesPerFile for item in sublist]
    return specImages


# -----------------------------------------------------------------------------
import torchvision.models as models

class BirdSoundResNet(nn.Module):
    def __init__(self, num_classes):
        super(BirdSoundResNet, self).__init__()
        # Set pretrained=False to avoid downloading weights if network issues occur
        self.base_model = models.resnet18(pretrained=False)
        in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

nClassesBirdClef2024 = 182
model = BirdSoundResNet(num_classes=nClassesBirdClef2024)
model.eval()  # Set model to evaluation mode

# -----------------------------------------------------------------------------
# predictions from the model
# -----------------------------------------------------------------------------
def getPredictions(specImages, model):
    rowIxs = list(range(len(specImages)))
    transforms_val = A.Compose([A.Normalize()])
    dataset = AudioDatasetTorchPrePro(rowIxs, specImages, transform=transforms_val, imageHeight=224, imageWidth=224)
    loader = DataLoader(dataset, batch_size=batchSizeForTesting, shuffle=False, num_workers=nWorkers, pin_memory=True)
    
    all_preds = np.empty((len(specImages), nClassesBirdClef2024), dtype=np.float32)
    with torch.no_grad():
        for batch in loader:
            inputs = batch['specImage']  # [B, C, H, W]
            row_indices = batch['rowIx']
            outputs = model(inputs)      # [B, nClasses]
            outputs = outputs.cpu().numpy()
            for i, row in enumerate(row_indices):
                all_preds[row] = outputs[i]
    return all_preds

# -----------------------------------------------------------------------------
def postProcess(preds, df, threshold=0.0, neighbouringAggregationFactor=1.0):
    print('Apply post processing', neighbouringAggregationFactor)
    filenames_unique = df.filename.unique()
    filenames = df.filename.tolist()
    for filename in filenames_unique:
        start_ix = filenames.index(filename)
        end_ix = len(filenames) - filenames[::-1].index(filename)
        preds_file = preds[start_ix:end_ix]
        if threshold:
            preds_file = preds_file * (preds_file >= threshold)
        next_preds = np.concatenate([preds_file[1:], np.zeros((1, preds_file.shape[-1]))])
        prev_preds = np.concatenate([np.zeros((1, preds_file.shape[-1])), preds_file[:-1]])
        preds[start_ix:end_ix] = preds_file + neighbouringAggregationFactor * next_preds + neighbouringAggregationFactor * prev_preds
    return preds


# -----------------------------------------------------------------------------
classIdsBirdClef2024 = ['asbfly', 'ashdro1', 'ashpri1', 'ashwoo2', 'asikoe2', 'asiope1', 'aspfly1', 'aspswi1',
                         'barfly1', 'barswa', 'bcnher', 'bkcbul1', 'bkrfla1', 'bkskit1', 'bkwsti', 'bladro1',
                         'blaeag1', 'blakit1', 'blhori1', 'blnmon1', 'blrwar1', 'bncwoo3', 'brakit1', 'brasta1',
                         'brcful1', 'brfowl1', 'brnhao1', 'brnshr', 'brodro1', 'brwjac1', 'brwowl1', 'btbeat1',
                         'bwfshr1', 'categr', 'chbeat1', 'cohcuc1', 'comfla1', 'comgre', 'comior1', 'comkin1',
                         'commoo3', 'commyn', 'compea', 'comros', 'comsan', 'comtai1', 'copbar1', 'crbsun2',
                         'cregos1', 'crfbar1', 'crseag1', 'dafbab1', 'darter2', 'eaywag1', 'emedov2', 'eucdov',
                         'eurbla2', 'eurcoo', 'forwag1', 'gargan', 'gloibi', 'goflea1', 'graher1', 'grbeat1',
                         'grecou1', 'greegr', 'grefla1', 'grehor1', 'grejun2', 'grenig1', 'grewar3', 'grnsan',
                         'grnwar1', 'grtdro1', 'gryfra', 'grynig2', 'grywag', 'gybpri1', 'gyhcaf1', 'heswoo1',
                         'hoopoe', 'houcro1', 'houspa', 'inbrob1', 'indpit1', 'indrob1', 'indrol2', 'indtit1',
                         'ingori1', 'inpher1', 'insbab1', 'insowl1', 'integr', 'isbduc1', 'jerbus2', 'junbab2',
                         'junmyn1', 'junowl1', 'kenplo1', 'kerlau2', 'labcro1', 'laudov1', 'lblwar1', 'lesyel1',
                         'lewduc1', 'lirplo', 'litegr', 'litgre1', 'litspi1', 'litswi1', 'lobsun2', 'maghor2',
                         'malpar1', 'maltro1', 'malwoo1', 'marsan', 'mawthr1', 'moipig1', 'nilfly2', 'niwpig1',
                         'nutman', 'orihob2', 'oripip1', 'pabflo1', 'paisto1', 'piebus1', 'piekin1', 'placuc3',
                         'plaflo1', 'plapri1', 'plhpar1', 'pomgrp2', 'purher1', 'pursun3', 'pursun4', 'purswa3',
                         'putbab1', 'redspu1', 'rerswa1', 'revbul', 'rewbul', 'rewlap1', 'rocpig', 'rorpar',
                         'rossta2', 'rufbab3', 'ruftre2', 'rufwoo2', 'rutfly6', 'sbeowl1', 'scamin3', 'shikra1',
                         'smamin1', 'sohmyn1', 'spepic1', 'spodov', 'spoowl1', 'sqtbul1', 'stbkin1', 'sttwoo1',
                         'thbwar1', 'tibfly3', 'tilwar1', 'vefnut1', 'vehpar1', 'wbbfly1', 'wemhar1', 'whbbul2',
                         'whbsho3', 'whbtre1', 'whbwag1', 'whbwat1', 'whbwoo2', 'whcbar1', 'whiter2', 'whrmun',
                         'whtkin2', 'woosan', 'wynlau1', 'yebbab1', 'yebbul3', 'zitcis1']
# 1. Prepare test DataFrame
test_df = get_test_df()

# 2. Extract spectrogram images for all segments
specImages = getSpecImages(test_df)
print('Spectrogram extraction complete. Total segments:', len(specImages))

# 3. Get predictions from the model
predictions = getPredictions(specImages, model)
print('predictions.shape', predictions.shape)

# 4. Optional postprocessing
if applylPostProcessing:
    predictions = postProcess(predictions, test_df, neighbouringAggregationFactor=neighbouringAggregationFactor)
print('predictions.shape after postprocessing', predictions.shape)

# 5. Create submission DataFrame and write CSV
prediction_df = pd.DataFrame(predictions, columns=classIdsBirdClef2024)
prediction_df.insert(loc=0, column='row_id', value=test_df['row_id'].tolist())
submission_path = 'submission.csv'
prediction_df.to_csv(submission_path, index=False)

print('Done.')

print("Script created. You can now run this script to generate your submission.csv file.")
