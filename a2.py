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

os.environ["MKL_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OMP_SCHEDULE"] = "STATIC"


inferenceStartTime = time.time()
checkTimePerBatchTime = 6000 # 100 min
checkTimePerBatch = False
maxInferenceTime = 7080 # 118 min
maxInferenceTimeReached = False

batchSizeForTesting = 8
nWorkers = None

applylPostProcessing = True
neighbouringAggregationFactor = 0.5

checkpointRootDir = '../input/bc24-2nd-place-models/'
dataDir = '../input/birdclef-2024/test_soundscapes/'

submissionMode = len(list(Path('../input/birdclef-2024/test_soundscapes/').glob('*.ogg'))) > 1


print('submissionMode', submissionMode)
print('dataDir', dataDir)


checkpoints = [
    {'path': '../input/rdc-birdclef/rdc-bird/ch0.pt'},
    {'path': '../input/rdc-birdclef/rdc-bird/ch1.pt'},
    {'path': '../input/rdc-birdclef/rdc-bird/ch2.pt'},
    {'path': '../input/rdc-birdclef/rdc-bird/ch3.pt'},
    {'path': '../input/rdc-birdclef/rdc-bird/ch4.pt'},
    {'path': '../input/rdc-birdclef/rdc-bird/ch5.pt'}
]


# Select checkpoints
cpIxs = [0]


checkpoints = [checkpoints[cpIx] for cpIx in cpIxs]
nCheckpoints = len(checkpoints)
print('nCheckpoints', nCheckpoints)


if not nWorkers:
    nWorkers = multiprocessing.cpu_count()


sampleRate = 32000
mel_spec_params = {
    "sample_rate": sampleRate,
    "n_mels": 64,
    "f_min": 20,
    "f_max": 16000,
    "n_fft": 2048,
    "hop_length": 1024,
    "normalized": True,
    "center" : True,
    "pad_mode" : "constant",
    "norm" : "slaney",
    "mel_scale" : "slaney"
}

top_db = 80


def normalize_melspec(X, eps=1e-6):
    mean = X.mean((1, 2), keepdim=True)
    std = X.std((1, 2), keepdim=True)
    Xstd = (X - mean) / (std + eps)

    norm_min, norm_max = (
        Xstd.min(-1)[0].min(-1)[0],
        Xstd.max(-1)[0].max(-1)[0],
    )
    fix_ind = (norm_max - norm_min) > eps * torch.ones_like(
        (norm_max - norm_min)
    )
    V = torch.zeros_like(Xstd)
    if fix_ind.sum():
        V_fix = Xstd[fix_ind]
        norm_max_fix = norm_max[fix_ind, None, None]
        norm_min_fix = norm_min[fix_ind, None, None]
        V_fix = torch.max(
            torch.min(V_fix, norm_max_fix),
            norm_min_fix,
        )
        V_fix = (V_fix - norm_min_fix) / (norm_max_fix - norm_min_fix)
        V[fix_ind] = V_fix
    return V


db_transform = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=top_db)

def getSpecImageTorch(sampleVec):

    # Convert sampleVec to torch tensor with shape (1, nFramesToRead)
    sampleVecTensor = torch.tensor(sampleVec, dtype=torch.float32).unsqueeze(0)

    melSpecDict = {}

    # torchMel_64_1024
    melSpecParams = mel_spec_params
    melSpecParams['n_mels'] = 64
    melSpecParams['hop_length'] = 1024
    melSpec = torchaudio.transforms.MelSpectrogram(**melSpecParams)
    melSpec = normalize_melspec(db_transform(melSpec(sampleVecTensor))) 
    melSpec = melSpec.squeeze(0).numpy()
    melSpecDict['torchMel_64_1024'] = melSpec

    # torchMel_128_512
    melSpecParams = mel_spec_params
    melSpecParams['n_mels'] = 128
    melSpecParams['hop_length'] = 512
    melSpec = torchaudio.transforms.MelSpectrogram(**melSpecParams)
    melSpec = normalize_melspec(db_transform(melSpec(sampleVecTensor)))
    melSpec = melSpec.squeeze(0).numpy()
    melSpecDict['torchMel_128_512'] = melSpec

    # torchMel_128_1024
    melSpecParams = mel_spec_params
    melSpecParams['n_mels'] = 128
    melSpecParams['hop_length'] = 1024
    melSpec = torchaudio.transforms.MelSpectrogram(**melSpecParams)
    melSpec = normalize_melspec(db_transform(melSpec(sampleVecTensor)))
    melSpec = melSpec.squeeze(0).numpy()
    melSpecDict['torchMel_128_1024'] = melSpec


    return melSpecDict


class AudioDatasetTorchPrePro(Dataset):
    
    def __init__(self, rowIxs, transform=None, preProType='torchMel_64_1024', imageHeight=64, imageWidth=128):
        self.rowIxs = rowIxs
        self.transform = transform
        self.preProType = preProType
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth

    def __len__(self):
        return len(self.rowIxs)

    def __getitem__(self, segmentIx):
        rowIx = self.rowIxs[segmentIx]

        
        specImage = specImages[rowIx][self.preProType]

        specImage = cv2.resize(specImage, (self.imageWidth, self.imageHeight), interpolation=cv2.INTER_LINEAR)
        
        specImage = (specImage * 255).astype(np.uint8)
        specImage = np.stack([specImage, specImage, specImage], axis=-1) # h,w --> h,w,3

        if self.transform: 
            res = self.transform(image=specImage)
            specImage = res['image']
        specImage = specImage.transpose(2, 0, 1) # h,w,c --> c,h,w
        return {'specImage': specImage, 'rowIx': rowIx}

def get_test_df():

    # Get dataframe from test files (filename, path, row_id, start_time, end_time)
    
    filenames = []
    paths = []
    row_ids = []
    start_times = []
    end_times = []
        
    files = os.listdir(dataDir)

    for file in files:
        if file.endswith('.ogg'):
            path = dataDir + file
            filename = os.path.splitext(file)[0]
            for end_time in range(5, 245, 5):
                row_id = filename + '_'  + str(end_time) 
                start_time = end_time - 5
                filenames.append(filename)
                paths.append(path)
                row_ids.append(row_id)
                start_times.append(start_time)
                end_times.append(end_time)

    test_df = pd.DataFrame({
            'filename': filenames,
            'path': paths,
            'row_id': row_ids,
            'start_time': start_times,
            'end_time': end_times
        })

    return test_df

def getSpecImagesPerFile(path, sampleRate=32000, segmentDuration=5.0):

    # Read audio file (assume mono, 32000 Hz)
    sampleVecFile, sampleRateSrc = sf.read(path)
    
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

    nFilesToProcess = len(paths)
    print('nFilesToProcess', nFilesToProcess)

    specImages = []
    fileIxStart = 0
    fileIxEnd = fileIxStart + nWorkers
    with Parallel(n_jobs=nWorkers) as parallel:

        while fileIxStart < nFilesToProcess:

            pathsToProcessInParallel = paths[fileIxStart:fileIxEnd]
            specImagesPerFile = parallel(delayed(getSpecImagesPerFile)(path) for path in pathsToProcessInParallel)
            # Flatten list
            specImagesPerFile = [item for sublist in specImagesPerFile for item in sublist]
            specImages += specImagesPerFile

            fileIxStart = fileIxEnd
            fileIxEnd = fileIxStart + nWorkers

    return specImages

def getModelAndConfigParams(checkpointDict):

    # Load model and cfg from torchscript checkpoint file
    extra_files = {'cfg.json': ''}
    model = torch.jit.load(checkpointDict['path'], _extra_files=extra_files)
    cfg = json.loads(extra_files['cfg.json'])

    print('Loaded:', cfg['runId'], cfg['preProType'], cfg['imageHeight'], cfg['imageWidth'])

    # Add model and cfg to checkpointDict
    checkpointDict['model'] = model
    checkpointDict['cfg'] = cfg

    return checkpointDict

def getPredictionsPerModel(rowIxs, nRowsTotal, checkpoints):

    global checkTimePerBatch
    global maxInferenceTimeReached

    nCheckpoints = len(checkpoints)
    predsPerModel = torch.full((nCheckpoints, nRowsTotal, nClassesBirdClef2024), -1, dtype=torch.float32)

    transforms_val = A.Compose([A.Normalize()])

    for cpIx in range(len(checkpoints)):

        inferenceStartTimePerModel = time.time()

        checkpoint = checkpoints[cpIx]
        model = checkpoint['model']

        preProType = checkpoint['cfg']['preProType']
        imageHeight = checkpoint['cfg']['imageHeight']
        imageWidth = checkpoint['cfg']['imageWidth']

        testDatasetDefault = AudioDatasetTorchPrePro(rowIxs, transform=transforms_val, preProType=preProType, imageHeight=imageHeight, imageWidth=imageWidth)
        testLoaderDefault = DataLoader(testDatasetDefault, batch_size=batchSizeForTesting, shuffle=False, num_workers=nWorkers, pin_memory=True)

        with torch.no_grad():
            for i, sample_batched in enumerate(testLoaderDefault):

                if checkTimePerBatch:
                    if (time.time() - inferenceStartTime) > maxInferenceTime:
                        maxInferenceTimeReached = True
                        print('maxInferenceTimeReached')
                        break

                input = sample_batched['specImage']
                rowIxsBatched = sample_batched['rowIx']
                output = model(input)
                predsPerModel[cpIx, rowIxsBatched] = output

        elapsedTimePerModel = time.time() - inferenceStartTimePerModel
        print(cpIx, checkpoint['cfg']['runId'], time.strftime("%H:%M:%S", time.gmtime(elapsedTimePerModel)))
        
        timeSinceInferenceStart = time.time() - inferenceStartTime
        
        if maxInferenceTimeReached or timeSinceInferenceStart > maxInferenceTime:
            maxInferenceTimeReached = True
            print('maxInferenceTimeReached')
            break

        if timeSinceInferenceStart > checkTimePerBatchTime:
            print('checkTimePerBatchTime reached')
            checkTimePerBatch = True


    predsPerModel = predsPerModel.numpy()

    return predsPerModel

def postProcess(preds, df, threshold=0.0, neighbouringAggregationFactor=1.0):
    print('Apply post processing', neighbouringAggregationFactor)
    filenames_unique = df.filename.unique()
    filenames = df.filename.tolist()
    for filename in filenames_unique:
        start_ix = filenames.index(filename)
        end_ix = len(filenames) - filenames[::-1].index(filename)
        preds_file = preds[start_ix:end_ix]
        if threshold:
            print('Threshold:', threshold)
            preds_file = preds_file * (preds_file >= threshold) # Remove preds < threshold
        # Predictions corresponding to prevoius/next window
        next_preds = np.concatenate([preds_file[1:], np.zeros((1, preds_file.shape[-1]))])
        prev_preds = np.concatenate([np.zeros((1, preds_file.shape[-1])), preds_file[:-1]])

        preds[start_ix:end_ix] = preds_file + neighbouringAggregationFactor * next_preds + neighbouringAggregationFactor * prev_preds  # Aggregating with neighbouring predictions

    return preds


# Main


classIdsBirdClef2024 = ['asbfly', 'ashdro1', 'ashpri1', 'ashwoo2', 'asikoe2', 'asiope1', 'aspfly1', 'aspswi1', 'barfly1', 'barswa', 'bcnher', 'bkcbul1', 'bkrfla1', 'bkskit1', 'bkwsti', 'bladro1', 'blaeag1', 'blakit1', 'blhori1', 'blnmon1', 'blrwar1', 'bncwoo3', 'brakit1', 'brasta1', 'brcful1', 'brfowl1', 'brnhao1', 'brnshr', 'brodro1', 'brwjac1', 'brwowl1', 'btbeat1', 'bwfshr1', 'categr', 'chbeat1', 'cohcuc1', 'comfla1', 'comgre', 'comior1', 'comkin1', 'commoo3', 'commyn', 'compea', 'comros', 'comsan', 'comtai1', 'copbar1', 'crbsun2', 'cregos1', 'crfbar1', 'crseag1', 'dafbab1', 'darter2', 'eaywag1', 'emedov2', 'eucdov', 'eurbla2', 'eurcoo', 'forwag1', 'gargan', 'gloibi', 'goflea1', 'graher1', 'grbeat1', 'grecou1', 'greegr', 'grefla1', 'grehor1', 'grejun2', 'grenig1', 'grewar3', 'grnsan', 'grnwar1', 'grtdro1', 'gryfra', 'grynig2', 'grywag', 'gybpri1', 'gyhcaf1', 'heswoo1', 'hoopoe', 'houcro1', 'houspa', 'inbrob1', 'indpit1', 'indrob1', 'indrol2', 'indtit1', 'ingori1', 'inpher1', 'insbab1', 'insowl1', 'integr', 'isbduc1', 'jerbus2', 'junbab2', 'junmyn1', 'junowl1', 'kenplo1', 'kerlau2', 'labcro1', 'laudov1', 'lblwar1', 'lesyel1', 'lewduc1', 'lirplo', 'litegr', 'litgre1', 'litspi1', 'litswi1', 'lobsun2', 'maghor2', 'malpar1', 'maltro1', 'malwoo1', 'marsan', 'mawthr1', 'moipig1', 'nilfly2', 'niwpig1', 'nutman', 'orihob2', 'oripip1', 'pabflo1', 'paisto1', 'piebus1', 'piekin1', 'placuc3', 'plaflo1', 'plapri1', 'plhpar1', 'pomgrp2', 'purher1', 'pursun3', 'pursun4', 'purswa3', 'putbab1', 'redspu1', 'rerswa1', 'revbul', 'rewbul', 'rewlap1', 'rocpig', 'rorpar', 'rossta2', 'rufbab3', 'ruftre2', 'rufwoo2', 'rutfly6', 'sbeowl1', 'scamin3', 'shikra1', 'smamin1', 'sohmyn1', 'spepic1', 'spodov', 'spoowl1', 'sqtbul1', 'stbkin1', 'sttwoo1', 'thbwar1', 'tibfly3', 'tilwar1', 'vefnut1', 'vehpar1', 'wbbfly1', 'wemhar1', 'whbbul2', 'whbsho3', 'whbtre1', 'whbwag1', 'whbwat1', 'whbwoo2', 'whcbar1', 'whiter2', 'whrmun', 'whtkin2', 'woosan', 'wynlau1', 'yebbab1', 'yebbul3', 'zitcis1']
nClassesBirdClef2024 = 182


# Get dataframe from test files (filename, path, row_id, start_time, end_time)
test_df = get_test_df()

# Get spec images for all file parts
specImages = getSpecImages(test_df)

# Get models and config params
for cp in checkpoints:
    cp = getModelAndConfigParams(cp)


# Get list of indices for all file parts
rowIxs = list(range(len(test_df)))
nRowsTotal = len(rowIxs)


elapsedTimeDataRead = time.time() - inferenceStartTime
print('ElapsedTimeDataRead [hh:mm:ss]: ', time.strftime("%H:%M:%S", time.gmtime(elapsedTimeDataRead)))

# Get predictions per model
predsPerModel = getPredictionsPerModel(rowIxs, nRowsTotal, checkpoints)
print('predsPerModel.shape', predsPerModel.shape)


# Mask missing values
predsPerModelMasked = np.ma.masked_equal(predsPerModel, -1)
# Average over checkpoints
predictions = np.mean(predsPerModelMasked, axis=0)


# Postprocessing
if applylPostProcessing:
    predictions = postProcess(predictions, test_df, neighbouringAggregationFactor=neighbouringAggregationFactor)


print('predictions.shape', predictions.shape)

# Convert to df
prediction_df = pd.DataFrame(predictions, columns=classIdsBirdClef2024)
prediction_df.insert(loc=0, column='row_id', value=test_df['row_id'].tolist())

# Write csv
prediction_df.to_csv('submission.csv', index=False)


elapsedTime = time.time() - inferenceStartTime
print('ElapsedTime [s]: ', elapsedTime)
print('ElapsedTime [hh:mm:ss]: ', time.strftime("%H:%M:%S", time.gmtime(elapsedTime)))
print('Done.')
