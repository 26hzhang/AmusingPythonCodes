from urllib.request import urlretrieve
import tarfile
import os
import pickle
import numpy as np
from tqdm import tqdm


class DownloadProgress(tqdm):
    last_block = 0

    def hook(self, block_num=1, block_size=1, total_size=None):
        self.total = total_size
        self.update((block_num - self.last_block) * block_size)
        self.last_block = block_num


def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def normalize(x):
    # min_val = np.min(x)
    # max_val = np.max(x)
    # x = (x - min_val) / (max_val - min_val)
    x = x / 255.0
    return x


def one_hot_encode(x):
    encoded = np.zeros((len(x), 10))
    for idx, val in enumerate(x):
        encoded[idx][val] = 1
    return encoded


def process_and_save(normalizer, one_hot_encoder, features, labels, filename):
    features = normalizer(features)
    labels = one_hot_encoder(labels)
    pickle.dump((features, labels), open(filename, 'wb'))


def load_cifar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    return features, labels


def preprocess_and_save_data(cifar10_dataset_folder_path, normalizer, one_hot_encoder, save_path):
    n_batches = 5
    valid_features = []
    valid_labels = []
    for batch_i in range(1, n_batches + 1):
        features, labels = load_cifar10_batch(cifar10_dataset_folder_path, batch_i)
        # find index to be the point as validation data in the whole dataset of the batch (10%)
        index_of_validation = int(len(features) * 0.1)
        process_and_save(normalizer, one_hot_encoder, features[:-index_of_validation], labels[:-index_of_validation],
                         save_path + 'batch_' + str(batch_i) + '.pkl')
        valid_features.extend(features[-index_of_validation:])
        valid_labels.extend(labels[-index_of_validation:])
    # preprocess the all stacked validation dataset
    process_and_save(normalizer, one_hot_encoder, np.array(valid_features), np.array(valid_labels),
                     save_path + 'valid.pkl')
    # load the test dataset
    with open(cifar10_dataset_folder_path + '/test_batch', mode='rb') as file:
        batch = pickle.load(file, encoding='latin1')
    # preprocess the testing data
    test_features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    test_labels = batch['labels']
    # Preprocess and Save all testing data
    process_and_save(normalizer, one_hot_encoder, np.array(test_features), np.array(test_labels),
                     save_path + 'test.pkl')


def maybe_download_and_extract():
    parent_folder = "./data/"
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    cifar10_dataset_zip_path = os.path.join(parent_folder, "cifar-10-python.tar.gz")
    # download the dataset (if not exist yet)
    if not os.path.isfile(cifar10_dataset_zip_path):
        with DownloadProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
            urlretrieve('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', cifar10_dataset_zip_path, pbar.hook)
    # extract file if not exist
    cifar10_dataset_folder_path = os.path.join(parent_folder, "cifar-10-batches-py")
    if not os.path.isdir(cifar10_dataset_folder_path):
        with tarfile.open(cifar10_dataset_zip_path) as tar:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=parent_folder)
            tar.close()
    # preprocess if not exist
    save_path = os.path.join(parent_folder, "cifar_pickle/")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path, "batch_1.pkl")):
        preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode, save_path)
    return save_path


def batch_features_labels(features, labels, batch_size):
    """Split features and labels into batches"""
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]


def load_training_batch(batch_id, batch_size):
    """Load the Preprocessed Training data and return them in batches of <batch_size> or less"""
    filename = 'data/cifar_pickle/' + 'batch_' + str(batch_id) + '.pkl'
    features, labels = pickle.load(open(filename, mode='rb'))
    return batch_features_labels(features, labels, batch_size)
