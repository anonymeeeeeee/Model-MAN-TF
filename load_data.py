from torch.utils.data import Dataset
import numpy as np
import scipy.io as sio
import copy
import utils
import math
from scipy.linalg import svd

#Centering: Subtracting the mean of all elements in the matrix (X - np.mean(X)) ensures that the matrix is centered around zero. This operation removes translation, making the data invariant to shifts in position.
#Scaling: Dividing the centered matrix by its Frobenius norm (X / normX) scales the matrix so that its Frobenius norm becomes 1. The Frobenius norm is a measure of the "size" of a matrix. Scaling to unit norm ensures that the matrix has a consistent scale across different instances.
#The resulting matrix after these operations is both centered (mean subtracted) and scaled (Frobenius norm equal to 1).

def CenteredScaled(X):
    # Reshape the input matrix to (n_frames, n_joints, k_dimensions)
    n_frames, total_dimensions = X.shape
    n_joints = total_dimensions // 3
    k_dimensions = 3
    X_reshaped = X.reshape((n_frames, n_joints, k_dimensions))
    X_reshaped = X_reshaped - np.mean(X_reshaped, axis=0)
    # Calculate the "centered" Frobenius norm for each joint
    normX = np.linalg.norm(X_reshaped, axis=(1, 2), ord='fro')
    # Scale to equal (unit) norm for each joint
    X_reshaped = X_reshaped / normX[:, np.newaxis, np.newaxis]
    # Reshape back to the original shape
    X_scaled = X_reshaped.reshape((n_frames, total_dimensions))
    return X_scaled


#Form prediction(test) data
class FormatDataPre(object):
    def __init__(self):
        pass
    def __call__(self, x_test, y_test):
        dec_in_test = x_test[-1:, :]
        x_test = x_test[:-1, :]
        return {'x_test': x_test, 'dec_in_test': dec_in_test, 'y_test': y_test}

#Form train/validation data
class FormatData(object):
    def __init__(self, config):
        self.config = config

    def __call__(self, sample, train):
        total_frames = self.config.input_window_size + self.config.output_window_size
        video_frames = sample.shape[0]
        idx = np.random.randint(1, video_frames - total_frames)
        data_seq = sample[idx:idx + total_frames, :]
        encoder_inputs = data_seq[:self.config.input_window_size - 1, :]
        if train:
            decoder_inputs = data_seq[self.config.input_window_size - 1:self.config.input_window_size - 1 + self.config.output_window_size, :]
        else:
            decoder_inputs = data_seq[self.config.input_window_size - 1:self.config.input_window_size, :]
        decoder_outputs = data_seq[self.config.input_window_size:, :]
        return {'encoder_inputs': encoder_inputs, 'decoder_inputs': decoder_inputs, 'decoder_outputs': decoder_outputs}

class LieTsfm(object):
    def __init__(self, config):
        self.config = config
    def __call__(self, sample):
        rawdata = sample
        data = rawdata[:, :-1, :3].reshape(rawdata.shape[0], -1)
        return data

class HumanDataset(Dataset):
    def __init__(self, config, train=True):
        self.config = config
        self.train = train
        self.lie_tsfm = LieTsfm(config)
        self.formatdata = FormatData(config)
        
        if config.datatype == 'lie':
            if train:
                train_path = './data/h3.6m/Train/train_lie'
            else:
                train_path = './data/h3.6m/Test/test_lie'
        elif config.datatype == 'xyz':
            if train:
                train_path = './data/h3.6m/Train/train_xyz'
            else:
                train_path = './data/h3.6m/Test/test_xyz'
        elif config.datatype == 'xyzl':
            if train:
                train_path = './data/h3.6m/Train/train_xyzl'
            else:
                train_path = './data/h3.6m/Test/test_xyzl'
        elif config.datatype == 'xyzk':
            if train:
                train_path = './data/h3.6m/Train/train_xyzk'
            else:
                train_path = './data/h3.6m/Test/test_xyzk'
                
        if train:
            subjects = ['S1', 'S6', 'S7', 'S8', 'S9', 'S11']
        else:
            subjects = ['S5']

        if config.filename == 'all':
            actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases', 'sitting','sittingdown', 'smoking','takingphoto', 'waiting', 'walking', 'walkingdog', 'walkingtogether']
        else:
            actions = [config.filename]

        set = []
        complete_train = []
        for id in subjects:
            for action in actions:
                for i in range(2):
                    if config.datatype == 'lie':
                        filename = '{0}/{1}_{2}_{3}_lie.mat'.format(train_path, id, action, i + 1)
                        rawdata = sio.loadmat(filename)['lie_parameters']
                        set.append(rawdata)
                    elif config.datatype in ['xyz', 'xyzl','xyzk']:
                        filename = '{0}/{1}_{2}_{3}_xyz.mat'.format(train_path, id, action, i + 1)
                        rawdata = sio.loadmat(filename)['joint_xyz']
                        set.append(rawdata.reshape(rawdata.shape[0], -1))

                if len(complete_train) == 0:
                    complete_train = copy.deepcopy(set[-1])
                else:
                    complete_train = np.append(complete_train, set[-1], axis=0)

        if not train and config.data_mean is None:
            print('Load train dataset first!')

        if train and config.datatype in ['xyz','lie','xyzl','xyzk']:
            data_mean, data_std, dim_to_ignore, dim_to_use = utils.normalization_stats(complete_train)
            config.data_mean = data_mean
            config.data_std = data_std
            config.dim_to_ignore = dim_to_ignore
            config.dim_to_use = dim_to_use
            
            set = utils.normalize_data(set, config.data_mean, config.data_std, config.dim_to_use)
                    
        # Apply CenteredScaled function only for 'xyz' datatype
        if config.datatype == 'xyzk':
            print("Centered Scaled process")
            set = [self.center_and_scale(skeleton) for skeleton in set]
            lie_tangent_space = None 
        
        self.data = set
        
    def center_and_scale(self, skeleton):
        skeleton = CenteredScaled(skeleton)
        return skeleton

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.config.datatype == 'lie':
            pass  # handle Lie data
        elif self.config.datatype == 'xyz':
            pass
        elif self.config.datatype == 'xyzk':
            pass
        elif self.config.datatype == 'xyzl':
            pass
        sample = self.formatdata(self.data[idx], False)
    
        return sample
    

class HumanPredictionDataset(object):
    def __init__(self, config):
        self.config = config
        if config.filename == 'all':
            self.actions = ['directions', 'discussion', 'eating', 'greeting', 'phoning', 'posing', 'purchases','sitting', 'sittingdown', 'smoking','takingphoto', 'waiting', 'walking', 'walkingdog','walkingtogether']
        else:
            self.actions = [config.filename]

        test_set = {}
        for subj in [5]:
            for action in self.actions:
                for subact in [1, 2]:
                    if config.datatype == 'lie':
                        filename = '{0}/S{1}_{2}_{3}_lie.mat'.format('./data/h3.6m/Test/test_lie', subj, action, subact)
                        test_set[(subj, action, subact)] = sio.loadmat(filename)['lie_parameters']

                    if config.datatype == 'xyz' or 'xyzl' or 'xyzk':
                        filename = '{0}/S{1}_{2}_{3}_xyz.mat'.format('./data/h3.6m/Test/test_xyz', subj, action, subact)
                        test_set[(subj, action, subact)] = sio.loadmat(filename)['joint_xyz']
                        test_set[(subj, action, subact)] = test_set[(subj, action, subact)].reshape(test_set[(subj, action, subact)].shape[0], -1)
        try:
            config.data_mean
        except NameError:
            print('Load train set first!')
        self.test_set = utils.normalize_data_dir(test_set, config.data_mean, config.data_std, config.dim_to_use)

    def get_data(self):
        x_test = {}
        y_test = {}
        dec_in_test = {}
        for action in self.actions:
            encoder_inputs, decoder_inputs, decoder_outputs = self.get_batch_srnn(self.config, self.test_set, action,self.config.output_window_size)
            x_test[action] = encoder_inputs
            y_test[action] = decoder_outputs
            dec_in_test[action] = np.zeros([decoder_inputs.shape[0], 1, decoder_inputs.shape[2]])
            dec_in_test[action][:, 0, :] = decoder_inputs[:, 0, :]
        return [x_test, y_test, dec_in_test]

    def get_batch_srnn(self, config, data, action, target_seq_len):
        # Obtain SRNN test sequences using the specified random seeds
        frames = {}
        frames[action] = self.find_indices_srnn(data, action)

        batch_size = 8
        subject = 5
        source_seq_len = config.input_window_size

        seeds = [(action, (i % 2) + 1, frames[action][i]) for i in range(batch_size)]

        encoder_inputs = np.zeros((batch_size, source_seq_len - 1, config.input_size), dtype=float)
        decoder_inputs = np.zeros((batch_size, target_seq_len, config.input_size), dtype=float)
        decoder_outputs = np.zeros((batch_size, target_seq_len, config.input_size), dtype=float)

        for i in range(batch_size):
            _, subsequence, idx = seeds[i]
            idx = idx + 50

            data_sel = data[(subject, action, subsequence)]
            data_sel = data_sel[(idx - source_seq_len):(idx + target_seq_len), :]

            encoder_inputs[i, :, :] = data_sel[0:source_seq_len - 1, :]  # x_test
            decoder_inputs[i, :, :] = data_sel[source_seq_len - 1:(source_seq_len + target_seq_len - 1),:]  # decoder_in_test
            decoder_outputs[i, :, :] = data_sel[source_seq_len:, :]  # y_test

        return [encoder_inputs, decoder_inputs, decoder_outputs]

    def find_indices_srnn(self, data, action):
        """
        Obtain the same action indices as in SRNN using a fixed random seed
        See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py
        """
        SEED = 1234567890
        rng = np.random.RandomState(SEED)

        subject = 5
        subaction1 = 1
        subaction2 = 2

        T1 = data[(subject, action, subaction1)].shape[0]
        T2 = data[(subject, action, subaction2)].shape[0]
        prefix, suffix = 50, 100

        idx = []
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))

        return idx

    def __len__(self):
        return len(self.data)
