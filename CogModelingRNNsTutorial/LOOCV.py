import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import os
from scipy.io import loadmat
import pickle
import disrnn
import subject_data
import rnn_utils
from rnn_utils import DatasetRNN

# Load data
#data_dir = '/usr/src/app/data/data_for_Meike.mat'
data_dir = '.\data\data_for_Meike.mat'
df = subject_data.load_data(data_dir)

# Create additional features
df['Reward'] = (df['select'] == 1) * df['Sacc'] + (df['select'] == -1) * df['Pacc']
df['theta'] = np.absolute(df['dtheta'])
df['theta_rescaled'] = df.groupby('subject')['theta'].transform(lambda x: pd.qcut(x, 4, labels=[0.2, 0.4, 0.6, 0.8]))
df['Sconfidence'] = df.groupby(['subject', 'block'])['Sreport'].transform(lambda x: np.where(x >= np.mean(x), 1, 0))
df['Pconfidence'] = df.groupby(['subject', 'block'])['Preport'].transform(lambda x: np.where(x >= np.mean(x), 1, 0))

# Define target and list of features from previous trial and current trial
features_previous_t = ['block', 'type', 'theta', 'Pconfidence', 'Pacc', 'Sacc', 'select', 'Reward', 'Sconfidence']
features_current_t = ['type', 'theta', 'Sacc']
target = ['Sconfidence']

categorical_partners = False
if categorical_partners:
    df, features_previous_t, features_current_t = subject_data.categorical_partners(df, features_previous_t, features_current_t)

# Shift features from previous trial up by one row
df_shifted, features_previous_t = subject_data.shift_df(df, features_previous_t)
features = features_previous_t + features_current_t
print(features)

df = pd.concat([df, df_shifted], axis=1)


def main():
    likelihoods = np.zeros((30,2))
    for i in range(df['subject'].unique().size):
        train_LL, test_LL = train_network_LOOCV(i)
        likelihoods[i,:] = [train_LL, test_LL]
    print(likelihoods)
    mean_LL_train = np.mean(likelihoods[:,0])
    mean_LL_test = np.mean(likelihoods[:,1])
    print(f'Mean LL train: {mean_LL_train:.3f}')
    print(f'Mean LL test: {mean_LL_test:.3f}')
    np.savetxt("likelihoods.csv", likelihoods, delimiter=",")

def train_network_LOOCV(leave_out_idx):
    print(f'Leave out subject: {leave_out_idx+1}')

    n_features = len(features)
    target_size = len(np.unique(df[target]))
    batch_size = None
    
    train, test = subject_data.train_test(df, features, target, leave_out_idx=leave_out_idx, batch_size=batch_size)

    # Set up the DisRNN
    latent_size = 8   
    obs_size = n_features
    update_mlp_shape = (n_features, n_features,)  
    choice_mlp_shape = (target_size,)

    def make_disrnn():
        model = disrnn.HkDisRNN(
        obs_size = obs_size,
        latent_size = latent_size,
        update_mlp_shape = update_mlp_shape,
        choice_mlp_shape = choice_mlp_shape,
        target_size=target_size)
        return model

    def make_disrnn_eval():
        model = disrnn.HkDisRNN(
            obs_size = obs_size,
            latent_size = latent_size,
            update_mlp_shape = update_mlp_shape,
            choice_mlp_shape = choice_mlp_shape,
            target_size=target_size,
            eval_mode=True)                  
        return model

    optimizer = optax.adam(learning_rate=1e-3)

    # Fit the model for a few steps without a penalty, to get a good starting point
    n_steps = 2000 #@param
    information_penalty = 0

    disrnn_params, opt_state, losses = rnn_utils.train_model(
        model_fun = make_disrnn,
        dataset = train,
        optimizer = optimizer,
        loss_fun = 'penalized_categorical',
        penalty_scale=information_penalty,
        n_steps=n_steps,
        do_plot=False,
        truncate_seq_length=200,
    )
           
    print(f'Normalized Likelihoods for Disentangled RNN, subject: {leave_out_idx+1}')
    print('Training Dataset')
    training_likelihood, training_output = subject_data.compute_log_likelihood(train, make_disrnn_eval, disrnn_params)
    print('Held-Out Dataset')
    testing_likelihood, test_output = subject_data.compute_log_likelihood(test, make_disrnn_eval, disrnn_params)
    
    # Save the model outputs
    if not os.path.exists('outputs_LOOCV'):
        os.makedirs('outputs_LOOCV')
    with open(f'outputs_LOOCV/outputs_test_{leave_out_idx+1}.pkl', 'wb') as f:
        pickle.dump(test_output, f)

    return training_likelihood, testing_likelihood


if __name__ == "__main__":
    main()

