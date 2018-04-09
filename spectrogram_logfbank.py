import csv
from python_speech_features import logfbank
import scipy.io.wavfile as wav

sample_rate, signal = scipy.io.wavfile.read('OSR_us_000_0010_8k.wav')  # File assumed to be in the same directory
signal = signal[0:int(3.5 * sample_rate)]  # Keep the first 3.5 seconds

logfbank_feat = logfbank(signal = signal,
                 samplerate = sample_rate,
                 winlen=0.025,
                 winstep=0.01, 
                 nfilt=40, 
                 nfft=512, 
                 lowfreq=0, 
                 highfreq = 4000,
                 preemph=0.97)

with open("fb_package.csv","w+") as my_csv:
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(logfbank_feat)
