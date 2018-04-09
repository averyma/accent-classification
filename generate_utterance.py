from pydub import AudioSegment
from pydub.silence import split_on_silence
from python_speech_features import fbank
from python_speech_features import delta
import scipy.io.wavfile
import csv
import numpy as np

# define input/output parameters
csv_file_path = "/home/averyma/accent-classification/cv-valid-train.csv"
audio_file_path = "/home/averyma/accent-classification/"
output_path = '/home/averyma/accent-classification/'
accent_label = 'us'
feature_type = 'mfsc'
section_len_threshold = 500 # only non-silent section>500ms will be considered
fs = 48000 # sample_rate
window_len = 7 # corresponds to a context window of 15 frames
accent_path = []
utterance = []

# read csv file
with open(csv_file_path, newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[6] == accent_label:
            accent_path.extend([audio_file_path + row[0]])
            
# pre-processing for each audio sample
for num, sample_path in enumerate(accent_path):
    # read audio sample specified by accent_path
    sample = AudioSegment.from_mp3(sample_path)

    print("Progress: " + str((num+1)/len(accent_path)*100) + "%")
    # "naive" silence removal approach -> threshold at -55 dB
    # divide the audio sample into non-silent sections
    sections = split_on_silence(sample,
                                min_silence_len=250,
                                silence_thresh=-55,
                                keep_silence=0)
    if sections:
        for section in sections: 
            # only work on section > 500ms
            if len(section) > section_len_threshold:
                # convert pydub.AudioSegment object to wav, which python_speech_feature packages uses
                wav = np.fromstring(section._data, dtype = np.int16)
                # extract static features:
                feature, energy = fbank(signal = wav, 
                                        samplerate = fs,
                                        winlen = 0.025, 
                                        winstep = 0.01,
                                        nfilt = 40, 
                                        nfft = 512,  
                                        lowfreq = 0, 
                                        highfreq = fs/2,
                                        preemph = 0.97,
                                        winfunc = np.hamming)
                # the feature here is size(frames, nfilt): (700,40)
                log_feature = np.log(feature) 
                # zero mean/unit variance
                static_feature = (log_feature - np.mean(log_feature, axis = 0))/np.std(log_feature,axis = 0)
                # extract delta of the static features
                d_feature = delta(static_feature,2)
                # extract delta-delta
                dd_feature = delta(d_feature,2)
                
                for i in range(window_len,len(static_feature)-window_len-1):
                    # concatenate into one utterance frame with size(nfilt, 3*context_window_len)
                    section_utterance = np.transpose(np.concatenate((static_feature[i-window_len:i+window_len+1],
                                                                    d_feature[i-window_len:i+window_len+1],
                                                                    dd_feature[i-window_len:i+window_len+1])))
                    # append to utterance list
                    utterance.append(np.int8(section_utterance))

# turn utterance list to np.ndarray
utterance = np.array(utterance)
# save utterance and label as .npy files 
np.save(output_path + accent_label + '_' + feature_type + '.npy',utterance)

