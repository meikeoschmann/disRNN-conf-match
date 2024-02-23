"""Functions for loading subject data."""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import jax

import rnn_utils
from rnn_utils import DatasetRNN

class SubjectData:

    def __init__(self, data, features_prev, features_curr, target, categorical_partners=False, categorical_theta=False):
        self.data = data
        self.features_prev = features_prev
        self.features_curr = features_curr
        self.target = target
        self.categorical_partners = categorical_partners
        self.categorical_theta = categorical_theta


        self.select_features()
        if self.categorical_partners:
            self.encode_partners()
        if self.categorical_theta:
            self.encode_theta()
        self.shift_df()
        
    def __call__(self):
        return self.data

    def select_features(self):
        '''Remove and create additional features for the dataset'''
        self.data['Reward'] = (self.data['select'] == 1) * self.data['Sacc'] + (self.data['select'] == -1) * self.data['Pacc']
        self.data['theta'] = np.absolute(self.data['dtheta'])
        self.data['Sconfidence'] = self.data.groupby(['subject', 'block'])['Sreport'].transform(lambda x: np.where(x >= np.mean(x), 1, 0))
        self.data['Pconfidence'] = self.data.groupby(['subject', 'block'])['Preport'].transform(lambda x: np.where(x >= np.mean(x), 1, 0))



    def encode_partners(self):
        '''Create one-hot encoding for block and partner type'''
        
        for i in range(1, 5):
            self.data['block' + str(i)] = self.data['block'].apply(lambda x: 1 if x == i else 0)
            for j in range(1, 3):
                self.data['b' + str(i) + '_p' + str(j)] = self.data['block' + str(i)] * self.data['type'].apply(lambda x: 1 if x == j else 0)
                self.features_prev.append('b' + str(i) + '_p' + str(j))
                self.features_curr.append('b' + str(i) + '_p' + str(j))
        self.features_prev.remove('type')
        self.features_curr.remove('type')

    # def encode_theta(data, features_prev, features_curr):
    #     '''Create one-hot encoding for theta values'''
        
    #     data['theta_1'] = data['theta_rescaled'].apply(lambda x: 1 if x == 0.2 else 0) # comparison with floats can cause problems
    #     data['theta_2'] = data['theta_rescaled'].apply(lambda x: 1 if x == 0.4 else 0)
    #     data['theta_3'] = data['theta_rescaled'].apply(lambda x: 1 if x == 0.6 else 0)
    #     data['theta_4'] = data['theta_rescaled'].apply(lambda x: 1 if x == 0.8 else 0)

    #     features_prev.extend(['theta_1', 'theta_2', 'theta_3', 'theta_4'])
    #     features_prev.remove('theta_rescaled')
    #     features_curr.extend(['theta_1', 'theta_2', 'theta_3', 'theta_4'])
    #     features_curr.remove('theta_rescaled')
        
    #     return data, features_prev, features_curr

    def shift_df(self): 
        '''Shift features of previous trial up by one row and rename the features. '''
        data_shifted = self.data.groupby(['subject', 'block'])[self.features_prev].shift(1, fill_value=0)
        self.features_prev = [str(col) + '_previous' for col in self.features_prev]
        data_shifted.columns = self.features_prev
        self.data = pd.concat([self.data, data_shifted], axis=1)
        

def train_test(df, features, target, leave_out_idx=0, batch_size=None):

    n_subjects = df['subject'].unique().size
    n_trials = int(len(df) / n_subjects)
    n_blocks = df['block'].unique().size
    n_features = len(features)

    xsTrain = np.zeros((n_trials, n_subjects, n_features))
    ysTrain = np.zeros((n_trials, n_subjects, 1))
    xsTest = np.zeros((n_trials, 1, n_features))
    ysTest = np.zeros((n_trials, 1, 1))

    for i, subject in enumerate(np.sort(df['subject'].unique())):
        xsTrain[:, i, :] = df[df['subject'] == subject][features].values
        ysTrain[:, i, :] = df[df['subject'] == subject][target].values

    # If subsequent encouter with same partner: Reshape target to fit the shape of ysTrain
    # ysTrain = np.array(target).reshape(n_trials, n_subjects, 1, order='F')

    # Experiment: Set SConfidence to 1 in test set
    #sconf_idx = features.index('Sconfidence_previous')
    #xsTest[:, 0, :] = xsTrain[:, leave_out_idx, :]
    #xsTest[:, 0, sconf_idx] = 1
    #ysTest[:, 0, :] = 1
    

    # The test set consists of the subject that is left out
    xsTest[:, 0, :] = xsTrain[:, leave_out_idx, :]
    ysTest[:, 0, :] = ysTrain[:, leave_out_idx, :]    
        
    # Exclude leave_out_idx from xsTrain and ysTrain
    xsTrain = np.delete(xsTrain, leave_out_idx, axis=1)
    ysTrain = np.delete(ysTrain, leave_out_idx, axis=1)
    n_subjects -= 1

    # division between blocks
    border = int(n_trials / len(df['block'].unique())) # number of trials per block
    indices = np.arange(border, n_blocks * border, border)

    # Add a dummy between the blocks
    xsTrain_padded_LOOCV = np.insert(xsTrain, indices, np.zeros((1, n_subjects, n_features)), axis=0)
    xsTest_padded_LOOCV = np.insert(xsTest, indices, np.zeros((1, 1, n_features)), axis=0)
    # The targets already contain -1s, but we need to add one more to seperate the blocks
    ysTrain_padded_LOOCV  = np.insert(ysTrain, indices, -1*np.ones((1, n_subjects, 1)), axis=0)
    ysTest_padded_LOOCV = np.insert(ysTest, indices, -1*np.ones((1, 1, 1)), axis=0)
    
    assert xsTrain_padded_LOOCV[64:-1:65, :, :].all() == 0, 'There should be a dummy between the blocks'
    assert xsTest_padded_LOOCV[64:-1:65, :, :].all() == 0, 'There should be a dummy between the blocks'
    assert np.unique(ysTrain_padded_LOOCV[64:-1:65, :, :]) == -1, 'There should be a dummy between the blocks'
    assert np.unique(ysTest_padded_LOOCV[64:-1:65, :, :]) == -1, 'There should be a dummy between the blocks'

    train = DatasetRNN(xsTrain_padded_LOOCV, ysTrain_padded_LOOCV, batch_size)
    test = DatasetRNN(xsTest_padded_LOOCV, ysTest_padded_LOOCV, batch_size)

    return train, test

def single_subject(df, features, target, subject_idx=1, batch_size=None):

    tr = 3
    df = df[df['subject'] == subject_idx]
    n_blocks = df['block'].unique().size
    n_trials = int(df['trial'].unique().size/n_blocks) 
    df_test = df[df['block'] > tr]
    df_train = df[df['block'] <= tr]
    n_features = len(features)


    xsTrain = np.zeros((n_trials, tr, n_features))
    ysTrain = np.zeros((n_trials, tr, 1))
    xsTest = np.zeros((n_trials, n_blocks-tr, n_features))
    ysTest = np.zeros((n_trials, n_blocks-tr, 1))

    for i in range(tr):
        xsTrain[:, i, :] = df_train[df_train['block'] == i+1][features].values
        ysTrain[:, i, :] = df_train[df_train['block'] == i+1][target].values

    for i in range(tr, n_blocks):
        xsTest[:, i-tr, :] = df_test[df_test['block'] == i+1][features].values
        ysTest[:, i-tr, :] = df_test[df_test['block'] == i+1][target].values
    
    train = DatasetRNN(xsTrain, ysTrain, batch_size)
    test = DatasetRNN(xsTest, ysTest, batch_size)

    return train, test
 

def compute_log_likelihood(dataset, model_fun, params):
  """Computes the log likelihood of the dataset under the model and the parameters.
  (the probability each choice we see in the dataset would have occurred in the model)
  
  Args:
    dataset: A DatasetRNN object.
    model_fun: A Haiku function that defines a network architecture.
    params: A set of params suitable for that network.
  """
  
  # It returns the normalized likelihood of the dataset under the model and the parameters as an output
  xs, actual_choices = next(dataset)
  n_trials_per_session, n_subjects = actual_choices.shape[:2]
  model_outputs, model_states = rnn_utils.eval_model(model_fun, params, xs)

  # Computes the logarithm of the softmax function, which rescales elements to the range [-infinity,0)
  predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :-1])) # last entry is nans
  
  log_likelihood = 0
  n = 0  # Total number of trials across sessions.
  for sess_i in range(n_subjects):
    for trial_i in range(n_trials_per_session):
      actual_choice = int(actual_choices[trial_i, sess_i]) # to match indices because choices are 1-6
      if actual_choice >= 0:  # values < 0 are invalid trials which we ignore.
        log_likelihood += predicted_log_choice_probabilities[trial_i, sess_i, actual_choice]
        n += 1

  normalized_likelihood = np.exp(log_likelihood / n)

  print(f'Normalized Likelihood: {100 * normalized_likelihood:.1f}%')

  return normalized_likelihood, model_outputs