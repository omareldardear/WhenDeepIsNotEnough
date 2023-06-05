
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
import os
from pathlib import Path



class audioCub_torchDataSet(Dataset):

    def __init__(self,
                 audio_dir,
                 transformation,
                 num_samples,
                 pd_data,
                 kind='train'):
        self.audio_dir = audio_dir
        self.num_samples = num_samples
        if (transformation):
            self.transformation = transformation

        if kind=='train':
          self.items = pd_data[pd_data['train'] == True][['recordedName','label','className']]
          self.items.reset_index(inplace = True,drop = True)
        elif kind=='test':
            self.items = pd_data[pd_data['train'] == False][['recordedName','label','className']]
            self.items.reset_index(inplace = True,drop = True)

        else:
          self.items =  pd_data[['recordedName','label','className']]
          self.items.reset_index(inplace = True,drop = True)

        self.length = len(self.items)

        self.targets = torch.as_tensor( list(self.items['label']))
    def __len__(self):
        return self.length


    def __getitem__(self, index):


        filename = self.items['recordedName'][index]
        className = self.items['className'][index]
        label = self.items['label'][index]
        signal, rate = torchaudio.load(self.audio_dir+ className + '/' + filename)
        signal = self._to_mono_if_necessary(signal)
        signal = self._cutting_if_necessary(signal)
        signal = self._padding_if_necessary(signal)
        data_tensor = self.transformation(signal)
        return (data_tensor, int(label))


    def _to_mono_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _cutting_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def _padding_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal


###########################################################################################

class ESC10_torchDataSet(Dataset):

    def __init__(self,
                 base_data_path,
                 transformation,
                 pd_data,
                 kind='train'):
        self.audio_dir = base_data_path
        if (transformation):
            self.transformation = transformation

        if kind=='train':
          self.items = pd_data[(pd_data['filename'].str.startswith('1-') ) |
                        (pd_data['filename'].str.startswith('2-'))  |
                        (pd_data['filename'].str.startswith('3-'))  |
                        (pd_data['filename'].str.startswith('4-'))][['filename','label']]
          self.items.reset_index(inplace = True,drop = True)
        elif kind=='val':
            self.items = pd_data[(pd_data['filename'].str.startswith('4-') )][['filename','label']]
            self.items.reset_index(inplace = True,drop = True)
        elif kind=='test':
            self.items =  pd_data[(pd_data['filename'].str.startswith('5-') )][['filename','label']]
            self.items.reset_index(inplace = True,drop = True)

        else:
          self.items =  pd_data[['filename','label']]
          self.items.reset_index(inplace = True,drop = True)

        self.length = len(self.items)

        self.targets = torch.as_tensor( list(self.items['label']))
    def __len__(self):
        return self.length


    def __getitem__(self, index):


        filename = self.items['filename'][index]
        label = self.items['label'][index]
        signal, rate = torchaudio.load(self.audio_dir+filename)
        signal = self._to_mono_if_necessary(signal)
        data_tensor = self.transformation(signal)
        return (data_tensor, int(label))


    def _to_mono_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

###########################################################################################


class ESC50_torchDataSet(Dataset):

    def __init__(self,
                 audio_dir,
                 transformation,
                 kind='train'):
        self.audio_dir = audio_dir
        if (transformation):
            self.transformation = transformation

        if kind=='train':
            files = Path(audio_dir).glob('[1-4]-*')
        elif kind=='val':
            files = Path(audio_dir).glob('4-*')
        elif kind=='test':
            files = Path(audio_dir).glob('5-*')
        else:
          files = Path(audio_dir).glob('*')


        self.items = [(str(file), file.name.split('-')[-1].replace('.wav', '')) for file in files]
        self.length = len(self.items)
        fileNameTuple, classIdTuple =zip(*self.items)
        self.targets = torch.as_tensor( list(map(int, classIdTuple)))
    def __len__(self):
        return self.length


    def __getitem__(self, index):


        filename, label = self.items[index]
        signal, rate = torchaudio.load(filename)
        signal = signal
        signal = self._to_mono_if_necessary(signal)
        data_tensor = self.transformation(signal)
        return (data_tensor, int(label))


    def _to_mono_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal




def loadDataSet(name="audioCub",
                setType="train",
                ballance=False,
                FRAME_SIZE = 1024,
                HOP_LENGTH = 512,
                N_MELS = 60):
    if(name=="audioCub"):

        BASE_DIR = ".."
        ANNOTATIONS_FILE = BASE_DIR + "/recordingInfo/validRecordings.csv"
        AUDIO_DIR = BASE_DIR + "/recordedAudio/"

        pd_data = pd.read_csv(ANNOTATIONS_FILE, index_col=0)
        # adding label column
        ord_enc = OrdinalEncoder()
        pd_data["label"] = ord_enc.fit_transform(pd_data[["className"]]).astype(int)

        # splittint train test
        pd_data['train'] = False
        trainPercent = 0.7
        for labelVal in pd_data.label.unique():
            trainCount = int(len(pd_data[pd_data.label == labelVal]) * trainPercent)
            pd_data.loc[pd_data[pd_data.label == labelVal].head(trainCount).index, 'train'] = True
        file = pd_data['recordedName'][0]
        className = pd_data['className'][0]
        fullName = AUDIO_DIR + className + '/' + file
        data_array, samplerate = sf.read(fullName)


        if(ballance):
            minClassSize = min([len(pd_data[pd_data['label'] == 0]), len(pd_data[pd_data['label'] == 1]),
                                len(pd_data[pd_data['label'] == 2]), len(pd_data[pd_data['label'] == 3])])
            for i in range(4):
                pd_data = pd_data.drop(pd_data[pd_data['label'] == i].index[minClassSize:])

            pd_data = pd_data.reset_index(drop=True)
            # splittint train test
            pd_data['train'] = False
            trainPercent = 0.7
            for labelVal in pd_data.label.unique():
                trainCount = int(len(pd_data[pd_data.label == labelVal]) * trainPercent)
                pd_data.loc[pd_data[pd_data.label == labelVal].head(trainCount).index, 'train'] = True


        SAMPLES_SIZE = int(5.8 * samplerate)
        SAMPLE_RATE = samplerate

        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=FRAME_SIZE,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            normalized=True
        )

        return audioCub_torchDataSet(AUDIO_DIR,
                               mel_spectrogram,
                               SAMPLES_SIZE,
                               pd_data,
                               kind=setType) , 4

    elif(name == "ESC10"):
        esc50_csv = '/home/icub/Documents/Omar/AudioCL/Datasets/ESC-50-master/meta/esc50.csv'
        base_data_path = '/home/icub/Documents/Omar/AudioCL/Datasets/ESC-50-master/audio/'
        pd_data = pd.read_csv(esc50_csv)
        pd_data = pd_data[pd_data['esc10'] == True]
        ord_enc = OrdinalEncoder()
        pd_data["label"] = ord_enc.fit_transform(pd_data[["category"]]).astype(int)
        file = '5-151085-A-20.wav'
        data_array, samplerate = sf.read(base_data_path + file)
        print('Sampling Rate:', samplerate)
        print('array', data_array)
        print('number of samples:', len(pd_data))
        print('number of wave files:', len(os.listdir(base_data_path)))

        SAMPLE_RATE = samplerate


        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=FRAME_SIZE,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            normalized=True
        )


        return ESC10_torchDataSet(base_data_path,
                               mel_spectrogram,
                               pd_data,
                               kind=setType) , 10

    elif (name == "ESC50"):

        esc50_csv = '/home/icub/Documents/Omar/AudioCL/Datasets/ESC-50-master/meta/esc50.csv'
        base_data_path = '/home/icub/Documents/Omar/AudioCL/Datasets/ESC-50-master/audio/'
        pd_data = pd.read_csv(esc50_csv)
        ord_enc = OrdinalEncoder()
        pd_data["label"] = ord_enc.fit_transform(pd_data[["category"]]).astype(int)
        file = '5-151085-A-20.wav'
        data_array, samplerate = sf.read(base_data_path + file)
        print('Sampling Rate:', samplerate)
        print('array', data_array)
        print('number of samples:', len(pd_data))
        print('number of wave files:', len(os.listdir(base_data_path)))

        SAMPLE_RATE = samplerate


        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=FRAME_SIZE,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS,
            normalized=True
        )

        return ESC50_torchDataSet(base_data_path,
                               mel_spectrogram,
                               kind=setType) , 50

    else:

        print("ERRORR ")
        return



classedCount = [4,10,50]
class Config(dict):
    def __getattribute__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, value):
        self[key] = value

    def print(self):
        print(self.keys())