"""Functions for loading subject data."""
import json
import os
import numpy as np
import pickle

from typing import List, Optional

# Do we even need a method for loading a single subject?
def load_data_for_one_subject(data, subjectID):
  """Load data for a single subject.

  Args:
    data: array of all the subjects
    subjectID: ID of the subject (1-30)

  Returns:
    subject_data: n_trials x n_sessions x 8 array of features
    """
  if subjectID is None:
    raise ValueError(f'Subject {subjectID} not found.')

  subject_xs = data[data[:,0,0] == subjectID, :, :]
  subject_ys = subject_xs[:, :, 6] # indx 6 is Sreport

  # Delete column containing subject IDs
  subject_xs = np.delete(subject_xs, 0, axis=2)
  
  # Check that the subject data has the right shape
  assert subject_xs.shape[1] == subject_ys.shape[1] == data.shape[1] # n_sessions
  assert subject_xs.shape[2] == data.shape[2]-1 # n_features-1
  n_subjects = np.unique(data[:,:,0]).size
  assert subject_xs.shape[0] == subject_ys.shape[0] == data.shape[0]/n_subjects # n_trials

  return subject_xs, subject_ys

def format_into_datasets(xs, ys, dataset_constructor):
  """Format inputs xs and outputs ys into dataset.

  Args:
    xs: n_trials x n_sessions x 8 array of features
    ys: n_trials x n_sessions  array of Sreport in subsequent trial
    dataset_constructor: constructor that accepts xs and ys as arguments; probably
      use rnn_utils.DatasetRNN

  Returns:
    dataset_train: a dataset containing even numbered sessions
    dataset_train: a dataset containing odd numbered sessions
  """
  n_sessions = xs.shape[1]
  n_features = xs.shape[2]

  # Add a dummy input at the beginning. First step has a target but no input
  xs = np.concatenate(
        (0. * np.ones((1,n_sessions,n_features)), xs), axis=0
    )
  
  # Add a dummy target at the end -- last step has input but no target
  ys = np.concatenate(
    (ys, -1*np.ones((1,n_sessions))), axis=0
    )
  
  n = int(xs.shape[1] // 2) * 2
  # even numbered sessions
  dataset_train = dataset_constructor(xs[:, :n:2], ys[:, :n:2]) # batchsize not specified!
  # odd numbered sessions
  dataset_test = dataset_constructor(xs[:, 1:n:2], ys[:, 1:n:2])
  
  return dataset_train, dataset_test


# def format_into_datasets(xs, ys, subjectID=None, dataset_constructor):
#   """Format inputs xs and outputs ys into dataset
#       and performs LOOCV for given subjectID.

#   Args:
#     xs: n_trials*n_subjects x n_sessions x 8 array of features
#     ys: n_trials x n_sessions x 1 array of public confidence in next trial
#     subjectID: ID of the subject to leave out(1-30)
#     dataset_constructor: constructor that accepts xs and ys as arguments; probably
#       use rnn_utils.DatasetRNN

#   Returns:
#     dataset_trains: a training dataset without subjectID
#     dataset_test: a testing dataset with only subjectID
#   """
  
#   # Add a dummy input at the beginning. First step has a target but no input
#   xs = np.concatenate(
#         (0. * np.ones(xs.shape), xs), axis=1
#     )
  
#   # Add a dummy target at the end -- last step has input but no target
#   ys = np.concatenate(
#     (ys, -1*np.ones(ys.shape)), axis=1
#     )
#   print(xs.shape)
#   print(ys.shape)
#   if subjectID != None:
#     # Perform LOOCV:
#     xs_train, xs_test = leave_one_out_cross_validation(subjectID, xs)
#     ys_train, ys_test = leave_one_out_cross_validation(subjectID, ys)

#   dataset_train = dataset_constructor(xs_train, ys_train)
#   dataset_test = dataset_constructor(xs_test, ys_test)
#   return dataset_train, dataset_test

def leave_one_out_cross_validation(subjectID, data):
    """Creates dataset for LOOCV, leaving out the specified subject.

  Args:
    subjectID: ID of the subject to leave out(1-30)
    data: array of all the subjects

  Returns:
    train_set: array without the subject
    test_set: array of only the subject
   """
    # Exclude the rows corresponding to the current subject
    # TO DO: make this nicer
    if data.ndim == 3: # for xs
      train_set = data[data[:, 0, 0] != subjectID]
      test_set = data[data[:, 0, 0] == subjectID]
    elif data.ndim == 1: # for ys
      train_set = data[data[:] != subjectID]
      test_set = data[data[:] == subjectID]
  
    return train_set, test_set
    
  


