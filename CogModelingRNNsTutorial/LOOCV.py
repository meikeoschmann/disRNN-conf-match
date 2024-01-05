import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from scipy.io import loadmat
import disrnn
import rnn_utils
from rnn_utils import DatasetRNN

matDf = loadmat('data\data_for_Meike.mat')['data']
features = matDf.dtype.names
df = pd.DataFrame(np.squeeze(np.array(matDf.tolist())).T, columns=features).drop(columns=['label','Spcorrect1', 'Spcorrect2', 'Snoise']).sort_values(by=['subject', 'trial'])
df['Reward'] = (df['select'] == 1) * df['Sacc'] + (df['select'] == -1) * df['Pacc']
features = ['dtheta', 'type', 'Pchoice', 'Preport', 'Pacc', 'Schoice', 'Sacc', 'Srt1', 'select', 'Reward', 'Sreport'] 
target = ['Sreport']


def main():
    likelihoods = np.zeros((30,2))
    for i in range(df['subject'].unique().size):
        train_LL, test_LL = train_network_LOOCV(i)
        likelihoods[i,:] = [train_LL, test_LL]
    print(likelihoods)
    np.savetxt("likelihoods.csv", likelihoods, delimiter=",")

def train_network_LOOCV(leave_out_idx):
    print(f'Leave out subject: {leave_out_idx+1}')
    batch_size = None
    n_trials = df['trial'].unique().size
    target_size = df['Sreport'].unique().size
    n_sessions = df['subject'].unique().size
    n_blocks = df['block'].unique().size
    features_number = len(features)

    xsTrain = np.zeros((n_trials, n_sessions, features_number))
    ysTrain = np.zeros((n_trials, n_sessions, 1))
    xsTest = np.zeros((n_trials, 1, features_number))
    ysTest = np.zeros((n_trials, 1, 1))

    for i, subject in enumerate(np.sort(df['subject'].unique())):
        xsTrain[:, i, :] = df[df['subject'] == subject][features].values
        ysTrain[:, i, :] = df[df['subject'] == subject][target].values
        xsTest[:, 0, :] = df[df['subject'] == subject][features].values[leave_out_idx,:]
        ysTest[:, 0, :] = df[df['subject'] == subject][target].values[leave_out_idx,:]
            
    # Exclude leave_out_idx from xsTrain and ysTrain
    xsTrain = np.delete(xsTrain, leave_out_idx, axis=1)
    ysTrain = np.delete(ysTrain, leave_out_idx, axis=1)

    n_sessions -= 1

    border = int(n_trials / len(df['block'].unique()))

    indices_xsTrain = np.arange(0, n_blocks * border, border)
    indices_xsTest = [0]
    indices_ysTrain = np.arange(border, n_blocks * border, border)
    
    # Add a dummy input at the beginning of each block. First step has a target but no input
    xsTrain_padded_LOOCV = np.insert(xsTrain, indices_xsTrain, np.zeros((1, n_sessions, features_number)), axis=0)
    xsTest_padded_LOOCV = np.insert(xsTest, indices_xsTest, np.zeros((1, 1, features_number)), axis=0)

    # Add a dummy target at the end of each block. Last step has input but no target
    ysTrain_padded_LOOCV  = np.insert(ysTrain, indices_ysTrain, -1*np.ones((1, n_sessions, 1)), axis=0)
    
    # np.insert inserts before given idx, so the last row we need to append manually
    ysTrain_padded_LOOCV = np.concatenate((ysTrain_padded_LOOCV, -1*np.ones((1,n_sessions,1))), axis=0)
    ysTest_padded_LOOCV = np.concatenate((ysTest, -1*np.ones((1,1,1))), axis=0)

    train = DatasetRNN(xsTrain_padded_LOOCV, ysTrain_padded_LOOCV, batch_size)
    test = DatasetRNN(xsTest_padded_LOOCV, ysTest_padded_LOOCV, batch_size)

    # Set up the DisRNN
    latent_size = 5    
    obs_size = xsTrain.shape[-1]
    update_mlp_shape = (5, 5, 5)  #@param
    choice_mlp_shape = (5, 5, 5)  #@param 

    #update_mlp_shape = (3,3,)  #@param
    # #@markdown Number of hidden units in each of the two layers of the choice MLP.
    #choice_mlp_shape = (2,)
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

    optimizer = optax.adam(learning_rate=1e-2)

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
        do_plot=True,
        truncate_seq_length=200,
    )
    # Now fit more steps with a penalty, to encourage it to find a simple solution
    # You can experiment with different values, but colab has been tested with 3000.
    n_steps = 3000  #@param
    information_penalty = 1e-3 #@param

    disrnn_params, opt_state, losses = rnn_utils.train_model(
        model_fun = make_disrnn,
        dataset = train,
        optimizer = optimizer,
        loss_fun = 'penalized_categorical',
        params=disrnn_params,
        opt_state=opt_state,
        penalty_scale=information_penalty,
        n_steps=n_steps,
        truncate_seq_length=200,
        do_plot=True,
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
        predicted_log_choice_probabilities = np.array(jax.nn.log_softmax(model_outputs[:, :, :-1])) # model_outputs[:,:,-1] is full of nans!

        log_likelihood = 0
        n = 0  # Total number of trials across sessions.
        for sess_i in range(n_sessions):
            for trial_i in range(n_trials_per_session):
                actual_choice = int(actual_choices[trial_i, sess_i])-1 # -1 because choices are between 1-6, to match the array indices
                if actual_choice >= 0:  # values < 0 are invalid trials which we ignore.
                    log_likelihood += predicted_log_choice_probabilities[trial_i, sess_i, actual_choice]
                    n += 1

        normalized_likelihood = np.exp(log_likelihood / n)

        print(f'Normalized Likelihood: {100 * normalized_likelihood:.1f}%')

        return normalized_likelihood
            
    print(f'Normalized Likelihoods for Disentangled RNN, subject: {leave_out_idx+1}')
    print('Training Dataset')
    training_likelihood = compute_log_likelihood(
    train, make_disrnn_eval, disrnn_params)
    print('Held-Out Dataset')
    testing_likelihood = compute_log_likelihood(
    train, make_disrnn_eval, disrnn_params)

    return training_likelihood, testing_likelihood


if __name__ == "__main__":
    main()

