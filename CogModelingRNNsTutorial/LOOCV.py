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
import rnn_utils
from rnn_utils import DatasetRNN

# If you use Sreport as a target, make sure to adapt the likelihood calculation to match the indices!

print('experiment with penalized categorical loss function, 2000 iterations with penalty 0, Sconf as last feature')
# Load data
matDf = loadmat('/usr/src/app/data/data_for_Meike.mat')['data']
features = matDf.dtype.names
df = pd.DataFrame(np.squeeze(np.array(matDf.tolist())).T, columns=features).drop(columns=['label','Spcorrect1', 'Spcorrect2', 'Snoise']).sort_values(by=['subject', 'trial'])

df['Reward'] = (df['select'] == 1) * df['Sacc'] + (df['select'] == -1) * df['Pacc']
df['theta'] = np.absolute(df['dtheta'])
df['theta_rescaled'] = df.groupby('subject')['theta'].transform(lambda x: pd.qcut(x, 4, labels=[0.2, 0.4, 0.6, 0.8]))

# Create Sconfidence for each subject and each block, relative to mean of Sreport: 1 = Sreport > mean, 0 = Sreport < mean
df['Sconfidence'] = df.groupby(['subject', 'block'])['Sreport'].transform(lambda x: np.where(x >= np.mean(x), 1, 0))
df['Pconfidence'] = df.groupby(['subject', 'block'])['Preport'].transform(lambda x: np.where(x >= np.mean(x), 1, 0))

# Create target: for a given index in df, find the next index that has the same value for df['type'] and add df['Sconfidence'] of that index to the target
target = []
Sacc_next = []
theta_next = []
# assert that target has the same length as df, i.e. that i == len(target)
for i in range(len(df)):
    for j in range(i+1, len(df)):
        if df['subject'][i] == df['subject'][j] and df['block'][i] == df['block'][j] and df['type'][i] == df['type'][j]:
            target.append(df['Sconfidence'][j])
            Sacc_next.append(df['Sacc'].iloc[j])
            theta_next.append(df['theta'].iloc[j])
            break

        # if the next trial is the first one of the next block, append -1
        elif df['subject'][i] == df['subject'][j] and (df['block'][i] + 1) == df['block'][j]:
            target.append(-1)
            Sacc_next.append(0)
            theta_next.append(0)
            break

        # if the next trial is the first one of the next subject, append -1
        elif (df['subject'][i] + 1) == df['subject'][j] and df['block'][i] == 4 and df['block'][j] == 1:
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

df['Sacc_next'] = Sacc_next
df['theta_next'] = theta_next

features = ['block', 'type', 'theta_rescaled', 'Pconfidence', 'Pacc', 'Schoice', 'Sacc', 'select', 'Reward', 'Sconfidence'] 
print(features)

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
    target_size = df['Sconfidence'].unique().size
    n_subjects = df['subject'].unique().size
    n_trials = int(len(df) / n_subjects)
    n_blocks = df['block'].unique().size
    n_features = len(features)
    batch_size = None
    
    xsTrain = np.zeros((n_trials, n_subjects, n_features))
    ysTrain = np.zeros((n_trials, n_subjects, 1))
    xsTest = np.zeros((n_trials, 1, n_features))
    ysTest = np.zeros((n_trials, 1, 1))

    for i, subject in enumerate(np.sort(df['subject'].unique())):
        xsTrain[:, i, :] = df[df['subject'] == subject][features].values
        
    # reshape target to fit the shape of ysTrain
    ysTrain = np.array(target).reshape(n_trials, n_subjects, 1, order='F')
    
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

    train = DatasetRNN(xsTrain_padded_LOOCV, ysTrain_padded_LOOCV, batch_size)
    test = DatasetRNN(xsTest_padded_LOOCV, ysTest_padded_LOOCV, batch_size)
    # batch_size = None
    # n_trials = 128
    # target_size = df['Sconfidence'].unique().size
    # n_sessions = df['subject'].unique().size
    # n_blocks = df['block'].unique().size
    # n_features = len(features)
  
    # xsTrain = np.zeros((n_trials, n_sessions, n_features ))
    # ysTrain = np.zeros((n_trials, n_sessions, 1))
    # xsTest = np.zeros((n_trials, 1, n_features ))
    # ysTest = np.zeros((n_trials, 1, 1))
    
    # for i, subject in enumerate(np.sort(df['subject'].unique())):
    #     xsTrain[:, i, :] = df[df['subject'] == subject][features].values
    #     ysTrain[:, i, :] = df[df['subject'] == subject][target].values
    #     xsTest[:, 0, :] = df[df['subject'] == subject][features].values[leave_out_idx,:]
    #     ysTest[:, 0, :] = df[df['subject'] == subject][target].values[leave_out_idx,:]

    # # Exclude leave_out_idx from xsTrain and ysTrain
    # xsTrain = np.delete(xsTrain, leave_out_idx, axis=1)
    # ysTrain = np.delete(ysTrain, leave_out_idx, axis=1)
    
    # n_sessions -= 1 
    
    # border = int(n_trials / len(df['block'].unique())) 
    
    # indices_xsTrain = np.arange(0, n_blocks * border, border) 
    # indices_xsTest = [0] 
    # indices_ysTrain = np.arange(border, n_blocks * border, border) 
        
    # # Add a dummy input at the beginning of each block. First step has a target but no input 
    # xsTrain_padded_LOOCV = np.insert(xsTrain, indices_xsTrain, np.zeros((1, n_sessions, n_features )), axis=0) 
    # xsTest_padded_LOOCV = np.insert(xsTest, indices_xsTest, np.zeros((1, 1, n_features )), axis=0) 
    
    # # Add a dummy target at the end of each block. Last step has input but no target 
    # ysTrain_padded_LOOCV  = np.insert(ysTrain, indices_ysTrain, -1*np.ones((1, n_sessions, 1)), axis=0) 
        
    # # np.insert inserts before given idx, so the last row we need to append manually 
    # ysTrain_padded_LOOCV = np.concatenate((ysTrain_padded_LOOCV, -1*np.ones((1,n_sessions,1))), axis=0) 
    # ysTest_padded_LOOCV = np.concatenate((ysTest, -1*np.ones((1,1,1))), axis=0) 
    
    # train = DatasetRNN(xsTrain_padded_LOOCV, ysTrain_padded_LOOCV, batch_size) 
    # test = DatasetRNN(xsTest_padded_LOOCV, ysTest_padded_LOOCV, batch_size) 


    # Save test dataset
    if not os.path.exists('test_datasets_LOOCV'):
        os.makedirs('test_datasets_LOOCV')
    with open(f'test_datasets_LOOCV/test_input_{leave_out_idx+1}.pkl', 'wb') as f:
        pickle.dump(xsTest_padded_LOOCV, f)
    with open(f'test_datasets_LOOCV/test_target_{leave_out_idx+1}.pkl', 'wb') as f:
        pickle.dump(ysTest_padded_LOOCV, f)

    # Set up the DisRNN
    latent_size = 5    
    obs_size = xsTrain.shape[-1]
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
        n_trials_per_session, n_sessions = actual_choices.shape[:2]
        model_outputs, model_states = rnn_utils.eval_model(model_fun, params, xs)

        # Computes the logarithm of the softmax function, which rescales elements to the range [-infinity,0)
        predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :-1])) # model_outputs[:,:,-1] is nan!

        log_likelihood = 0
        n = 0  # Total number of trials across sessions.
        for sess_i in range(n_sessions):
            for trial_i in range(n_trials_per_session):
                actual_choice = int(actual_choices[trial_i, sess_i]) # -1 if Sreport is target because choices are between 1-6, to match the array indices
                if actual_choice >= 0:  # values < 0 are invalid trials which we ignore.
                    log_likelihood += predicted_log_choice_probabilities[trial_i, sess_i, actual_choice]
                    n += 1

        normalized_likelihood = np.exp(log_likelihood / n)

        print(f'Normalized Likelihood: {100 * normalized_likelihood:.1f}%')

        return normalized_likelihood, model_outputs
            
    print(f'Normalized Likelihoods for Disentangled RNN, subject: {leave_out_idx+1}')
    print('Training Dataset')
    training_likelihood, training_output = compute_log_likelihood(train, make_disrnn_eval, disrnn_params)
    print('Held-Out Dataset')
    testing_likelihood, test_output = compute_log_likelihood(test, make_disrnn_eval, disrnn_params)
    
    # Save the model outputs
    if not os.path.exists('outputs_LOOCV'):
        os.makedirs('outputs_LOOCV')
    with open(f'outputs_LOOCV/outputs_test_{leave_out_idx+1}.pkl', 'wb') as f:
        pickle.dump(test_output, f)

    return training_likelihood, testing_likelihood


if __name__ == "__main__":
    main()

