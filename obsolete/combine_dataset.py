# The original Common Voice speech corpus provided on Kaggle has a very poor
# train/dev/test split (ie: 97%,1.5%,1.5%).
# Therefore, I wrote a script to combine the extracted features together.
# However, considering the dev/test data only contribute to a very insignificant
# amount compared to the total data. This step is no longer needed, meaning I simply
# split the cv-valid-train dataset into 70%:15%:15% and go from there.

import numpy as np

lan = 'us'
output_lan = 'us'
file_path = '/home/averyma/accent-classification/mfsc/'

train = np.load(file_path + lan + '_mfsc_train.npy')
dev = np.load(file_path + lan + '_mfsc_dev.npy')
test = np.load(file_path + lan + '_mfsc_test.npy')
combine = np.concatenate((train,dev,test))

print('train.shape: ' + str(train.shape))
print('dev.shape: ' + str(dev.shape))
print('test.shape: ' + str(test.shape))
print('combine.shape: ' + str(combine.shape))

np.save('/home/averyma/accent-classification/mfsc/' + output_lan + '_mfsc.npy',combine)







