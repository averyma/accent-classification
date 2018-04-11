import numpy as np

# read dataset
print('reading dataset...')
data1 = np.load('/home/averyma/accent-classification/mfsc/us_mfsc.npy')
data2 = np.load('/home/averyma/accent-classification/mfsc/uk_mfsc.npy')
data1 = data1[-len(data2):]
data = np.concatenate((data1, data2))

# create labels
print('creating labels...')
num_class = 2
label1 = np.array(np.ones(len(data1))*0, dtype = int)
label2 = np.array(np.ones(len(data2))*1, dtype = int)
label = np.concatenate((label1, label2))
# one_hot = np.zeros((len(label), num_class))
# one_hot[np.arange(len(label)), label] = 1

# shuffle dataset and labels
print('shuffling...')
idx = np.random.permutation(len(data))
data, label = data[idx], label[idx]

# split test/dev/train set: 70%/15%/15% for both data and labels
print('spliting...')
num_dev = int(len(data)*.15)
dev_data = data[0:num_dev,:]
test_data = data[num_dev:2*num_dev,:]
train_data = data[2*num_dev:]
dev_label = label[0:num_dev]
test_label = label[num_dev:2*num_dev]
train_label = label[2*num_dev:]

# save data and labels
print('saving...')
output_path = '/home/averyma/accent-classification/mfsc/us_uk_mfsc/'
np.save(output_path + 'test_data.npy', test_data)
np.save(output_path + 'dev_data.npy', dev_data)
np.save(output_path + 'train_data.npy', train_data)
np.save(output_path + 'test_label.npy', test_label)
np.save(output_path + 'dev_label.npy', dev_label)
np.save(output_path + 'train_label.npy', train_label)

