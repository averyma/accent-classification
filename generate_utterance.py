import csv
from pydub import AudioSegment
from pydub.silence import split_on_silence
from python_speech_features import mfcc
from python_speech_features import delta
import scipy.io.wavfile

# define input/output parameters
csv_file_path = "/Users/ama/git-repo/accent-classification/dummy.csv"
audio_file_path = "/Users/ama/git-repo/accent-classification/"
output_path = '/Users/ama/git-repo/accent-classification/'
accent_label = 'us'
feature_type = 'mfcc'
section_len_threshold = 500 # only non-silent section>500ms will be considered
fs = 48000 # sample_rate
window = 7 # corresponds to a context window of 15 frames
accent_path = []
utterance = []

# read csv file
with open(csv_file_path, newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[6] == accent_label:
            accent_path.extend([audio_file_path + row[0]])
    
# read in all mp3 files, path specified by accent_path
playlist = [AudioSegment.from_mp3(mp3_file) for mp3_file in accent_path]

# pre-processing for each audio sample
for sample in playlist:
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
                static_feature = mfcc(signal = wav, 
                                  samplerate = fs,
                                  winlen = 0.025, 
                                  winstep = 0.01,
                                  nfilt = 40, 
                                  nfft = 512,  
                                  lowfreq = 0, 
                                  highfreq = fs/2,
                                  numcep = 40, 
                                  ceplifter = 22, 
                                  preemph = 0.97,
                                  appendEnergy = True,
                                  winfunc = numpy.hamming)
                # extract delta of the static features
                d_feature = delta(static_feature,2)
                # extract delta-delta
                dd_feature = delta(d_feature,2)
                
                # concatenate into one utterance frame with size(numcep, 3*context_window)
                section_utterance = np.transpose(np.concatenate((static_feature[i-window:i+window+1],
                                                                d_feature[i-window:i+window+1],
                                                                dd_feature[i-window:i+window+1])))
                # append to utterance list
                utterance.append(section_utterance)

# turn utterance list to np.ndarray
utterance = np.array(utterance)
# generate labels: (us:1, uk:2, ind:3, aus:4, can:5)
label = np.ones(len(utterance),dtype=int8) * 1
# save utterance and label as .npy files 
np.save(output_path + accent_label + '_' + feature_type + '.npy',utterance)
np.save(output_path + accent_label + '_label.npy',label)
