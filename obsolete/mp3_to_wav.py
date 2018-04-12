# Convert mp3 audio data to much preferred wav files.
# No longer needed since it was handled during the generate_utterance stgage.

import csv
from pydub import AudioSegment

with open('dummy.csv', newline='') as f:
    reader = csv.reader(f)
    filenames = [row[0] for row in reader]
    filenames = filenames[1:]
#     print(filenames)
    
    for file in filenames:
        sample_mp3 = AudioSegment.from_mp3(file)
        sample_mp3.export("wav/sample-" + str(file[21:-5]) + ".wav", format="wav")
