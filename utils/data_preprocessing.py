import os
import pickle
import numpy as np

def load_batch(file_path):
    with open(file_path, 'rb') as file:
        batch = pickle.load(file, encoding='latin1')
    data = batch['data']
    labels = batch['labels']
    data = data.reshape((10000, 3, 32, 32))
    data = data.transpose(0, 2, 3, 1)
    labels = np.array(labels)
    return data, labels

def load_data(data_dir):
    train_data = []
    train_labels = []
    for i in range(1, 6):
        data, labels = load_batch(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(data)
        train_labels.append(labels)
    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)
    
    test_data, test_labels = load_batch(os.path.join(data_dir, 'test_batch'))
    
    return (train_data, train_labels), (test_data, test_labels)

def preprocess_data(train_images, test_images):
    train_images = train_images.reshape((50000, 32, 32, 3))
    test_images = test_images.reshape((10000, 32, 32, 3))
    train_images = train_images.astype('float32')/255
    test_images = test_images.astype('float32')/255
    
    return train_images, test_images