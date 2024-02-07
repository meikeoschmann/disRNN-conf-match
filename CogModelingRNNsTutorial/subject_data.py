"""Functions for loading subject data."""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat
import jax

import rnn_utils
from rnn_utils import DatasetRNN

DATA_DIR = '..\data\data_for_Meike.mat'

def load_data(data_dir=DATA_DIR):
    '''Load subject data from matlab file.'''

    if not os.path.exists(data_dir):
        raise ValueError('Data directory does not exist.')

    matDf = loadmat(data_dir)['data']
    features = matDf.dtype.names
    df = df = pd.DataFrame(np.squeeze(np.array(matDf.tolist())).T, columns=features).drop(columns=['label','Spcorrect1', 'Spcorrect2', 'Snoise']).sort_values(by=['subject', 'trial'])

    return df


def categorical_partners(data, features_prev, features_curr):
    '''Create one-hot encoding for block and partner type'''
    
    data['block1'] = data['block'].apply(lambda x: 1 if x == 1 else 0)
    data['block2'] = data['block'].apply(lambda x: 1 if x == 2 else 0)
    data['block3'] = data['block'].apply(lambda x: 1 if x == 3 else 0)
    data['block4'] = data['block'].apply(lambda x: 1 if x == 4 else 0)
    data['b1_p1'] = data['block1'] * data['type'].apply(lambda x: 1 if x == 1 else 0)
    data['b1_p2'] = data['block1'] * data['type'].apply(lambda x: 1 if x == 2 else 0)
    data['b2_p1'] = data['block2'] * data['type'].apply(lambda x: 1 if x == 1 else 0)
    data['b2_p2'] = data['block2'] * data['type'].apply(lambda x: 1 if x == 2 else 0)
    data['b3_p1'] = data['block3'] * data['type'].apply(lambda x: 1 if x == 1 else 0)
    data['b3_p2'] = data['block3'] * data['type'].apply(lambda x: 1 if x == 2 else 0)
    data['b4_p1'] = data['block4'] * data['type'].apply(lambda x: 1 if x == 1 else 0)
    data['b4_p2'] = data['block4'] * data['type'].apply(lambda x: 1 if x == 2 else 0)

    features_prev.extend(['b1_p1', 'b1_p2', 'b2_p1', 'b2_p2', 'b3_p1', 'b3_p2', 'b4_p1', 'b4_p2'])
    features_prev.remove('type')
    features_curr.extend(['b1_p1', 'b1_p2', 'b2_p1', 'b2_p2', 'b3_p1', 'b3_p2', 'b4_p1', 'b4_p2'])
    features_curr.remove('type')
    
    return data, features_prev, features_curr

def categorical_theta(data, features_prev, features_curr):
    '''Create one-hot encoding for theta values'''
    
    data['theta_1'] = data['theta_rescaled'].apply(lambda x: 1 if x == 0.2 else 0)
    data['theta_2'] = data['theta_rescaled'].apply(lambda x: 1 if x == 0.4 else 0)
    data['theta_3'] = data['theta_rescaled'].apply(lambda x: 1 if x == 0.6 else 0)
    data['theta_4'] = data['theta_rescaled'].apply(lambda x: 1 if x == 0.8 else 0)

    features_prev.extend(['theta_1', 'theta_2', 'theta_3', 'theta_4'])
    features_prev.remove('theta_rescaled')
    features_curr.extend(['theta_1', 'theta_2', 'theta_3', 'theta_4'])
    features_curr.remove('theta_rescaled')
    
    return data, features_prev, features_curr


def target_next_encounter(data):
    '''Create target based on next encounter with the same partner: 
    for a given index in the dataframe, find the next index that has the same value for data['type']
    and add data['Sconfidence'] of that index to the target list'''
    
    subject_list = list(data['subject'].unique())
    target = []
    Sacc_next = []
    theta_next = []
    for i in range(len(data)):
        for j in range(i+1, len(data)):
            if data['subject'].iloc[i] == data['subject'].iloc[j] and data['block'].iloc[i] == data['block'].iloc[j] and data['type'].iloc[i] == data['type'].iloc[j]:
                target.append(data['Sconfidence'].iloc[j])
                Sacc_next.append(data['Sacc'].iloc[j])
                theta_next.append(data['theta'].iloc[j])
                break

            # if the next trial is the first one of the next block, append -1
            elif data['subject'].iloc[i] == data['subject'].iloc[j] and (data['block'].iloc[i] + 1) == data['block'].iloc[j]:
                target.append(-1)
                Sacc_next.append(0)
                theta_next.append(0)
                break

            # if the next trial is the first one of the next subject, append -1
            elif (subject_list.index(data['subject'].iloc[i])+1 < len(subject_list)) and (subject_list[subject_list.index(data['subject'].iloc[i])+1]) == data['subject'].iloc[j] and data['block'].iloc[i] == 4 and data['block'].iloc[j] == 1:
                target.append(-1)
                Sacc_next.append(0)
                theta_next.append(0)
                break
    # no target for the last two trials, append -1 twice    
    target.append(-1)
    target.append(-1)
    Sacc_next.append(0)
    Sacc_next.append(0)
    theta_next.append(0)
    theta_next.append(0)

    data['Sacc_next'] = Sacc_next
    data['theta_next'] = theta_next

    return data, target

def shift_df(df, features_prev): 
    '''Shift features of previous trial up by one row and rename the features. '''
    df = df.groupby(['subject', 'block'])[features_prev].shift(1, fill_value=0)
    df.columns = [str(col) + '_previous' for col in df.columns]
    features_prev = [str(col) + '_previous' for col in features_prev]
    return df, features_prev


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
  #predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :-1])) # last entry is nans
  predicted_log_choice_probabilities = np.log(model_outputs[:, :, :-1]) # last entry is nans
  
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