'''
Training and testing the GRU model to compensate for the disturbances.

Author: Shrenik Zinage, Vrushabh Zinage
'''

from src.GRUNetwork import GRUNetwork
from src.functions import create_sequences, dataloader, attitude_dynamics, euler_rates, psd_from_fft
from src.PIDController import PIDController

# Required libraries
import jax                                          # JAX library for accelerated numerical computing
import jax.numpy as jnp                             # NumPy-like API for JAX
import jax.lax as lax                               # Linear algebra library for JAX
import jax.random as jrandom                        # Random number generation for JAX
import numpy as np                                  # NumPy library for numerical computing
import optax                                        # Optimization library for JAX
import equinox as eqx                               # Deep learning library for JAX
from scipy.io import loadmat, savemat               # Functions for reading and writing MATLAB files
from sklearn.preprocessing import StandardScaler    # Standardize features by removing the mean and scaling to unit variance
import matplotlib.pyplot as plt                     # Plotting library
from typing import List, Tuple                      # Type hints for function signatures
import pandas as pd                                 # Data manipulation library
from numpy import linalg as LA                      # Linear algebra library
import time                                         # Time-related functions
import seaborn as sns                               # Data visualization library based on matplotlib
import matplotlib.lines as mlines

sns.set_context("paper")
sns.set_style("ticks")
plt.rcParams['text.latex.preamble'] = r"\usepackage{lmodern}"
params = {'text.usetex' : True,
          'font.size' : 12,
          'font.family' : 'lmodern'
          }
plt.rcParams.update(params)

# Parameters
batch_size = 64                 # Batch size for training
epochs = 500                    # Number of epochs for training
learning_rate = 5e-3            # Learning rate for the optimizer
sequence_length = 5             # Input window size for the GRU model
hidden_sizes=[128, 128, 128]    # Hidden sizes for the GRU model
seed = 5678   
patience = 30                   # Early stopping patience
train_iter = 5                  # Total number of times of retraining the model


def huber_loss(pred_y, y, delta=1.0):
    residual = jnp.abs(pred_y - y)
    loss = jnp.where(residual < delta, 0.5 * residual**2, delta * (residual - 0.5 * delta))
    return loss

@eqx.filter_value_and_grad
def compute_loss(model, x, y):

    pred_y = jax.vmap(model)(x)
    return jnp.mean(huber_loss(pred_y, y))

# Important for efficiency whenever you use JAX: wrap everything into a single JIT region.
@eqx.filter_jit
def make_step(model, x, y, opt_state):

    loss, grads = compute_loss(model, x, y)
    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)

    return loss, model, opt_state

iterations = 5  # The attitude error time series data is divided into 5 parts i.e. 86400/5

norm_error=[]
tt=[]

means_attitude = np.empty((3,1))
stds_attitude = np.empty((3,1))

means_euler = np.empty((3,1))
stds_euler = np.empty((3,1))

for jj in range(train_iter):

    start_time_session = time.time()

    for ii in range(iterations-1):

        # error1.mat contains the 24 hrs (86400s) time series data of attitude (actual disturbances)
        filename = f"data/d_actual{ii}.mat"

        if ii==0:

            data = loadmat(filename)['error1'].T

        else:

            data = loadmat(filename)['y_test_pred'].T


        scaler = StandardScaler()

        # Normalize the data
        normalized_data = scaler.fit_transform(data)

        # The training size takes on 1/(iterations-ii) portion of the original time series data set
        train_size = int(len(normalized_data) * (1-((iterations-ii-1)/(iterations-ii))))

        # The test size contains the remaining portion
        test_size = len(normalized_data) - train_size

        train_data = normalized_data[:train_size]
        test_data = normalized_data[train_size:]

        X_train, y_train = create_sequences(train_data, sequence_length)
        X_test, y_test = create_sequences(test_data, sequence_length)

        data_key, model_key = jrandom.split(jrandom.PRNGKey(seed), 2)
        iter_data = dataloader((X_train, y_train), batch_size)

        model = GRUNetwork(in_size = 1, out_size = 1, hidden_sizes = hidden_sizes, key = model_key)

        losses = []

        optim = optax.adam(learning_rate)
        opt_state = optim.init(model)

        best_loss = float('inf')
        epochs_no_improve = 0

        start_time_iteration = time.time()

        for step, (x, y) in zip(range(epochs), iter_data):

            start_time = time.time()

            loss, model, opt_state = make_step(model, x, y, opt_state)

            end_time = time.time()
            epoch_time = end_time - start_time

            loss = loss.item()
            losses.append(loss)

            if loss < best_loss:
                    best_loss = loss
                    epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve == patience:
                # print(f'Early stopping on epoch {step}')
                break

            # print(f"Epoch={step}, loss={loss}, time={epoch_time} seconds")
        
        end_time_iteration = time.time()
        print(f"Iteration {ii + 1} of train iteration {jj+1} took {end_time_iteration - start_time_iteration} seconds")

        # Plot the loss curve
        plt.figure(figsize=(4, 2.5), dpi=200)
        plt.plot(losses)
        plt.xlabel('Epoch')
        plt.ylabel('Training Loss')
        plt.title('Iteration '+ str(ii+1))
        plt.grid(alpha = 0.8)
        plt.savefig(f"figures/loss_curve{ii}.pdf", format='pdf', bbox_inches='tight', dpi=100)
        plt.show()

        test_predictions = jax.vmap(model)(X_test)
        test_predictions = scaler.inverse_transform(test_predictions)

        # These are the predictions of GRU over the test data i.e. (1-1/(iterations-ii)) portion of the overall data
        y_test_pred = np.array(test_predictions.reshape(-1))

        # These test predictions are stored in d_predictions{ii}.mat
        filename = f"data/d_predictions{ii}.mat"
        savemat(filename, {'y_test_pred': y_test_pred})

        # Initialize Euler angles and their log
        euler_angles = np.array([0.0000001, 0.0000001, 0.0000001])  # Initial angles: phi, theta, psi
        euler_log = []
        controls_log=[]

        # Load disturbances from d_actual{ii}.mat which are the actual disturbances
        filename=f"data/d_actual{ii}.mat"
        mat_data = loadmat(filename)

        if ii==0:

            disturbances = mat_data['error1'].squeeze()

        else:

            disturbances = mat_data['y_test_pred'].squeeze()

        # Load the disturbance predictions from the trained GRU model
        filename = f"data/d_predictions{ii}.mat"
        mat_data = loadmat(filename)

        disturbances_estimate = mat_data['y_test_pred'].squeeze()

        frac=1/iterations

        time_steps=int((1-(ii+1)/iterations)*86400)

        dt = 0.1

        moments_of_inertia = [38.33, 345.0, 300.0]  # Dummy values; adjust as necessary

        # Initialize PID controllers for p, q, r rates
        pid_controllers = {axis: PIDController(Kp=1, Ki=0.1, Kd=0.01, set_point=0) for axis in ('p', 'q', 'r')}
        pid_p, pid_q, pid_r = pid_controllers['p'], pid_controllers['q'], pid_controllers['r']  

        # Initial attitude rates
        attitude_rates = np.array([2e-7, 2e-7, 2e-7])

        # Log for simulation
        attitudes_log = []

        for t in range(min(time_steps,len(disturbances_estimate))):

            control_input_p = pid_p.compute(attitude_rates[0])
            control_input_q = pid_q.compute(attitude_rates[1])
            control_input_r = pid_r.compute(attitude_rates[2])

            control_input = np.array([control_input_p, control_input_q, control_input_r])

            rate_dots, controls = attitude_dynamics(attitude_rates, moments_of_inertia, control_input, disturbances[t+int(frac*86400)-1],disturbances_estimate[t])
            # rate_dots = attitude_dynamics(attitude_rates, moments_of_inertia, control_input, disturbances[t+int(frac*86400)-1],disturbances[t+int(frac*86400)-1])

            for k in range(10):

                attitude_rates = attitude_rates + rate_dots * dt

                # Update Euler angles
                euler_dot = euler_rates(attitude_rates, euler_angles)
                # for l in range(10):
                euler_angles = euler_angles + euler_dot * dt

            attitudes_log.append(attitude_rates.copy())
            euler_log.append(euler_angles.copy())
            controls_log.append(controls.copy())


        euler_log=np.array(euler_log)
        controls_log=np.array(controls_log)
        attitudes_log = np.array(attitudes_log)
        norm_error.append(LA.norm(euler_log))
        tt.append(ii)

        # Compute the mean along the rows
        mean_attitude = np.mean(attitudes_log, axis=0).reshape((3,1))

        # Compute the standard deviation along the rows
        std_attitude = np.std(attitudes_log, axis=0).reshape((3,1))

        mean_euler = np.mean(euler_log, axis=0).reshape((3,1))

        # Compute the standard deviation along the rows
        std_euler = np.std(euler_log, axis=0).reshape((3,1))

        means_attitude=np.append(means_attitude,mean_attitude,axis=1)
        stds_attitude=np.append(stds_attitude,std_attitude,axis=1)
        means_euler=np.append(means_euler,mean_euler,axis=1)
        stds_euler=np.append(stds_euler,std_euler,axis=1)

        euler_log=np.array(euler_log)
        y_test_pred=attitudes_log[:,0]
        y_test_pred=np.transpose(y_test_pred)

        filename=f"data/d_actual{ii+1}.mat"
        savemat(filename, {'y_test_pred': y_test_pred})

        offset=2500

        t=np.linspace(ii*frac*86400+offset,ii*frac*86400+min(frac*86400,len(attitudes_log)), min(int(frac*86400),len(attitudes_log))-offset)

        t = np.arange(0, 1478*4, 0.1)

        # Opacity for different legends
        opacity = [0.5, 0.75, 1.0]

        data_points = 14780 if ii <= 2 else 14770

        # Plot the angular rates
        plt.figure(figsize=(5, 3), dpi=200)
        line1 = plt.plot(t[iterations : data_points + iterations], attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 0]/1e-6, label='p', color='blue', alpha = opacity[0],)
        line2 = plt.plot(t[iterations : data_points + iterations], attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 1]/1e-6, label='q', color='green', alpha = opacity[1], linestyle='dashed')
        line3 = plt.plot(t[iterations : data_points + iterations], attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 2]/1e-6, label='r', color='red', alpha = opacity[2], linestyle='dotted')
        plt.xlabel('Time (s)', fontsize=10, labelpad=10)
        plt.ylabel(r'Angular rates ($10^{-6}$ rad/s)', fontsize = 10, labelpad = 10)
        plt.title('Iteration ' + str(ii+1))
        legend_handles = [mlines.Line2D([], [], color='blue', alpha=opacity[0], label='p'),
                        mlines.Line2D([], [], color='green', alpha=opacity[1], label='q', linestyle='dashed'),
                        mlines.Line2D([], [], color='red', alpha=opacity[2], label='r', linestyle='dotted')]
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
        plt.xticks(np.arange(0, 2000, step=500))
        plt.grid(True, linestyle=':', linewidth=0.5, color='gray')
        sns.despine(trim=True)
        plt.tight_layout()
        filename=f"figures/angular_rates_plot{ii}.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=400)
        plt.show()

        # Plot the Euler angles
        plt.figure(figsize=(5, 3), dpi=200)
        line1 = plt.plot(t[iterations : data_points + iterations], euler_log[0+offset:int(frac*86400), 0]/4.84814e-6, label=r'$\phi$', color='blue', alpha = opacity[0])
        line2 = plt.plot(t[iterations : data_points + iterations], euler_log[0+offset:int(frac*86400), 1]/4.84814e-6, label=r'$\theta$', color='green', alpha = opacity[1], linestyle='dashed')
        line3 = plt.plot(t[iterations : data_points + iterations], euler_log[0+offset:int(frac*86400), 2]/4.84814e-6, label=r'$\psi$', color='red', alpha = opacity[2], linestyle='dotted')
        plt.xlabel('Time (s)', fontsize=10, labelpad=10)
        plt.ylabel(r'Euler angles (arcsec)', fontsize=10, labelpad=10)
        plt.title('Iteration ' + str(ii+1))
        legend_handles = [mlines.Line2D([], [], color='blue', alpha=opacity[0], label=r'$\phi$'),
                        mlines.Line2D([], [], color='green', alpha=opacity[1], label=r'$\theta$', linestyle='dashed'),
                        mlines.Line2D([], [], color='red', alpha=opacity[2], label=r'$\psi$', linestyle='dotted')]
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
        plt.xticks(np.arange(0, 2000, step=500))
        plt.grid(True, linestyle=':', linewidth=0.5, color='gray')
        sns.despine(trim=True)
        plt.tight_layout()
        filename=f"figures/euler_angles_plot{ii}.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=400)
        plt.show()

                                                    ###### PSD plots #####

        m = 100             # hamming variable
        time_step = 0.1      # time step

        s_psd1, omega_psd1 = psd_from_fft(attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 0]/1e-6, len(attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 0]), m, time_step)
        s_psd2, omega_psd2 = psd_from_fft(attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 1]/1e-6, len(attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 1]), m, time_step)
        s_psd3, omega_psd3 = psd_from_fft(attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 2]/1e-6, len(attitudes_log[0+offset:min(int(frac*86400),len(attitudes_log)), 2]), m, time_step)

        s_psd4, omega_psd4 = psd_from_fft(euler_log[0+offset:min(int(frac*86400),len(attitudes_log)), 0]/4.84814e-6, len(euler_log[0+offset:min(int(frac*86400),len(euler_log)), 0]), m, time_step)
        s_psd5, omega_psd5 = psd_from_fft(euler_log[0+offset:min(int(frac*86400),len(attitudes_log)), 1]/4.84814e-6, len(euler_log[0+offset:min(int(frac*86400),len(euler_log)), 1]), m, time_step)
        s_psd6, omega_psd6 = psd_from_fft(euler_log[0+offset:min(int(frac*86400),len(attitudes_log)), 2]/4.84814e-6, len(euler_log[0+offset:min(int(frac*86400),len(euler_log)), 2]), m, time_step)

        plt.figure(figsize=(5, 3), dpi=200)
        line1 = plt.plot(omega_psd1, s_psd1, label='p', color='blue', alpha = opacity[0],)
        line2 = plt.plot(omega_psd2, s_psd2, label='q', color='green', alpha = opacity[1], linestyle='dashed')
        line3 = plt.plot(omega_psd3, s_psd3, label='r', color='red', alpha = opacity[2], linestyle='dotted')
        plt.xlabel('Frequency (rad/s)', fontsize=10, labelpad=10)
        # plt.ylabel(r'PSD of angular rates ($10^{-12} (rad/s)^2/Hz)$)', fontsize = 8, labelpad = 10)
        plt.ylabel('Spectral density function of angular rates' + '\n' + r'($10^{-12}$ (rad/s)$^2$/Hz)', fontsize = 10, labelpad = 10)
        plt.title('Iteration ' + str(ii+1))
        legend_handles = [mlines.Line2D([], [], color='blue', alpha=opacity[0], label='p'),
                        mlines.Line2D([], [], color='green', alpha=opacity[1], label='q', linestyle='dashed'),
                        mlines.Line2D([], [], color='red', alpha=opacity[2], label='r', linestyle='dotted')]
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle=':', linewidth=0.5, color='gray')
        plt.xlim(0, 1)
        sns.despine(trim=True)
        plt.tight_layout()
        filename=f"figures/psd_angular_rates_plot_{ii}.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=400)
        plt.show()

        plt.figure(figsize=(5, 3), dpi=200)
        line1 = plt.plot(omega_psd4, s_psd4, label=r'$\phi$', color='blue', alpha = opacity[0])
        line2 = plt.plot(omega_psd5, s_psd5, label=r'$\theta$', color='green', alpha = opacity[1], linestyle='dashed')
        line3 = plt.plot(omega_psd6, s_psd6, label=r'$\psi$', color='red', alpha = opacity[2], linestyle='dotted')
        plt.xlabel('Frequency (rad/s)', fontsize=10, labelpad=10)
        plt.ylabel('Spectral density function of euler angles' + '\n' + r'(arcsec$^2$/Hz)', fontsize = 10, labelpad = 10)
        plt.title('Iteration ' + str(ii+1))
        legend_handles = [mlines.Line2D([], [], color='blue', alpha=opacity[0], label=r'$\phi$'),
                        mlines.Line2D([], [], color='green', alpha=opacity[1], label=r'$\theta$', linestyle='dashed'),
                        mlines.Line2D([], [], color='red', alpha=opacity[2], label=r'$\psi$', linestyle='dotted')]
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, linestyle=':', linewidth=0.5, color='gray')
        plt.xlim(0, 1)
        sns.despine(trim=True)
        plt.tight_layout()
        filename=f"figures/psd_euler_angles_plot_{ii}.pdf"
        plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=400)
        plt.show()


        end_index = min(int(frac*86400), len(attitudes_log))
        y_data_lines = [attitudes_log[0+offset:end_index, i]/1e-6 for i in range(3)]
        y_data_lines += [euler_log[0+offset:end_index, i]/4.84814e-6 for i in range(3)]

        # Calculate the square root of the sum of squares and divide by the total number of points
        results = [np.sqrt(np.sum(np.square(line))) / len(line) for line in y_data_lines]

        results_dict = {}

        if ii == 0:
            globals()[f'results_1_{jj+1}'] = results
        if ii == 1:
            globals()[f'results_2_{jj+1}'] = results
        if ii == 2:
            globals()[f'results_3_{jj+1}'] = results
        if ii == 3:
            globals()[f'results_4_{jj+1}'] = results
    
    end_time_session = time.time()
    print(f"Training and implementation session {jj+1} took {end_time_session - start_time_session} seconds")

                                            ###### Box Plot vs iterations #####

for k in range(6):

        data = []

        for i in range(1, 5):
            data.append([locals()[f'results_{i}_{j}'][k] for j in range(1, 6)])

        plt.figure(figsize=(5, 3), dpi=200)
        plt.boxplot(data, labels=[str(i) for i in range(1, 5)], showfliers=False)
        # plt.yscale('log')
        plt.xlabel('Iterations', fontsize=10, labelpad=10)
        plt.grid(True, linestyle=':', linewidth=0.5, color='gray')
        sns.despine(trim=True)
        plt.tight_layout()

        if k == 0:

            plt.ylabel(r'$p$ ($10^{-12}$ rad/s)', fontsize = 10, labelpad = 10)
            plt.savefig('figures/box_plot_p.pdf', format='pdf', bbox_inches='tight', dpi=400)

        if k == 1:

            plt.ylabel(r'$q$ ($10^{-12}$ rad/s)', fontsize = 10, labelpad = 10)
            plt.savefig('figures/box_plot_q.pdf', format='pdf', bbox_inches='tight', dpi=400)

        if k == 2:

            plt.ylabel(r'$r$ ($10^{-12}$ rad/s)', fontsize = 10, labelpad = 10)
            plt.savefig('figures/box_plot_r.pdf', format='pdf', bbox_inches='tight', dpi=400)

        if k == 3:

            plt.ylabel(r'$\phi$ (arcsec)', fontsize = 10, labelpad = 10)
            plt.savefig('figures/box_plot_phi.pdf', format='pdf', bbox_inches='tight', dpi=400)

        if k == 4:

            plt.ylabel(r'$\theta$ (arcsec)', fontsize = 10, labelpad = 10)
            plt.savefig('figures/box_plot_theta.pdf', format='pdf', bbox_inches='tight', dpi=400)
        
        if k == 5:

            plt.ylabel(r'$\psi$ (arcsec)', fontsize = 10, labelpad = 10)
            plt.savefig('figures/box_plot_psi.pdf', format='pdf', bbox_inches='tight', dpi=400)

        plt.show()

                                            ###### RMSE vs iterations #####

results = [[results_1_1, results_1_2, results_1_3, results_1_4, results_1_5],
           [results_2_1, results_2_2, results_2_3, results_2_4, results_2_5],
           [results_3_1, results_3_2, results_3_3, results_3_4, results_3_5],
           [results_4_1, results_4_2, results_4_3, results_4_4, results_4_5]]

rmse_errors = np.array([[np.mean([result[i] for result in results_set]) for i in range(6)] for results_set in results])

tt = np.array([1, 2, 3, 4])     # Iterations

# Plot RMSE of angular rates vs iterations
plt.figure(figsize=(5, 3), dpi=200)
plt.plot(tt, rmse_errors[:,0], 'o-', label='p', color='blue')
plt.plot(tt, rmse_errors[:,1], 'o-', label='q', color='green')
plt.plot(tt, rmse_errors[:,2], 'o-', label='r', color='red')
plt.xlabel('Iterations', fontsize=10, labelpad=10)
plt.ylabel('Mean RMSE of angular rates' + '\n' + r'($10^{-6}$ rad/s)', fontsize = 10, labelpad = 10)
plt.legend()
plt.xticks(np.arange(0, 6, step=1))
plt.xlim(0.8, 4.2)
plt.grid(True, linestyle=':', linewidth=0.5, color='gray')
sns.despine(trim=True)
plt.tight_layout()
filename=f"figures/rmse_vs_iterations_angular.pdf"
plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=400)
plt.show()

# Plot RMSE of euler angles vs iterations
plt.figure(figsize=(5, 3), dpi=200)
plt.plot(tt, rmse_errors[:,3], 'o-', label=r'$\phi$', color='blue')
plt.plot(tt, rmse_errors[:,4], 'o-', label=r'$\theta$', color='green')
plt.plot(tt, rmse_errors[:,5], 'o-', label=r'$\psi$', color='red')
plt.xlabel('Iterations', fontsize=10, labelpad=10)
plt.ylabel('Mean RMSE of euler angles' + '\n' + r'(arcsec)', fontsize = 10, labelpad = 10)
plt.legend()
plt.xticks(np.arange(0, 6, step=1))
plt.xlim(0.8, 4.2)
plt.grid(True, linestyle=':', linewidth=0.5, color='gray')
sns.despine(trim=True)
plt.tight_layout()
filename=f"figures/rmse_vs_iterations_euler.pdf"
plt.savefig(filename, format='pdf', bbox_inches='tight', dpi=400)
plt.show()
