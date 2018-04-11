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







