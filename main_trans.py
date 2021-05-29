import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import stats
import pickle
import time
import sys
import matplotlib.pyplot as plt
import joblib
import keras
from bayes_opt import BayesianOptimization

from data_preprocessing import get_spikes_with_history

from metrics import get_R2
from metrics import get_rho

import transformer
from decoders import GRUDecoder, LSTMDecoder, CNNDecoder

############### PARAMETER SETTING ###############
#'s1': Somatosensory Cortex
#'m1': Motor Cortex
#'hc': Hippocampus
type = 'hc'
mode = 'cross validation'

############### DATA PROCESS ###############
# Import data
folder = '/content/drive/MyDrive/Decoding_Data/'
with open(folder + 'example_data_'+type+'.pickle', 'rb') as f: 
    if type == 'hc':
        neural_data, pos_binned = pickle.load(f, encoding='latin1')
    else:
        neural_data, vels_binned = pickle.load(f, encoding='latin1')

# Choose how many bins for one prediction 
if type == 's1':
    bins_before = 6
    bins_current = 1
    bins_after = 6
if type == 'm1':
    bins_before = 13
    bins_current = 1
    bins_after = 0
if type == 'hc':
    bins_before = 4
    bins_current = 1
    bins_after = 5

# Define input and output of our model
if type == 'hc':
    y = pos_binned
else:
    y = vels_binned
# Remove neurons with too few spikes in HC dataset
if type == 'hc':
    nd_sum = np.nansum(neural_data, axis=0)
    rmv_nrn = np.where(nd_sum < 100)
    neural_data = np.delete(neural_data, rmv_nrn, 1)
# Format for recurrent networks
x = get_spikes_with_history(neural_data, bins_before, bins_after, bins_current)
# Remove time bins with no output (y value)
if type == 'hc':
    rmv_time = np.where(np.isnan(y[:, 0]) | np.isnan(y[:, 1]))
    x = np.delete(x, rmv_time, 0)
    y = np.delete(y, rmv_time, 0)
    x = x[: int(.8 * x.shape[0]), :, :]
    y = y[: int(.8 * y.shape[0]), :]
# s1: x(61339, 13, 52) y(61339, 2)
# m1: x(25299, 14, 164) y(25299, 2)
# hc: x(22283, 10, 46) y(22283, 2)

# Train/Val/Test split
valid_range_all=[[0,.1],[.1,.2],[.2,.3],[.3,.4],[.4,.5],[.5,.6],[.6,.7],[.7,.8],[.8,.9],[.9,1]]
testing_range_all=[[.1,.2],[.2,.3],[.3,.4],[.4,.5],[.5,.6],[.6,.7],[.7,.8],[.8,.9],[.9,1],[0,.1]]
training_range_all=[[[.2,1]],[[0,.1],[.3,1]],[[0,.2],[.4,1]],[[0,.3],[.5,1]],[[0,.4],[.6,1]],
            [[0,.5],[.7,1]],[[0,.6],[.8,1]],[[0,.7],[.9,1]],[[0,.8]],[[.1,.9]]]
num_folds=len(valid_range_all)

# R2 values
mean_r2=np.empty(num_folds)
# Actual data
y_test_all=[]
y_train_all=[]
y_valid_all=[]
# Test predictions
y_pred_all=[]
# Training predictions
y_train_pred_all=[]
# Validation predictions
y_valid_pred_all=[]


t1=time.time() 

num_examples=x.shape[0]

############### TRAIN/VAL/TEST ###############
for i in range(num_folds):

  ##### DATASPLIT ######
  # Get testing set for this fold
  testing_range=testing_range_all[i]
  testing_set=np.arange(np.int(np.round(testing_range[0]*num_examples))+bins_before,np.int(np.round(testing_range[1]*num_examples))-bins_after)

  # Get validation set for this fold
  valid_range=valid_range_all[i]
  valid_set=np.arange(np.int(np.round(valid_range[0]*num_examples))+bins_before,np.int(np.round(valid_range[1]*num_examples))-bins_after)

  # Get training set for this fold. 
  training_ranges=training_range_all[i]
  for j in range(len(training_ranges)): 
      training_range=training_ranges[j]
      if j==0: 
          training_set=np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,np.int(np.round(training_range[1]*num_examples))-bins_after)
      if j==1: 
          training_set_temp=np.arange(np.int(np.round(training_range[0]*num_examples))+bins_before,np.int(np.round(training_range[1]*num_examples))-bins_after)
          training_set=np.concatenate((training_set,training_set_temp),axis=0)
                
  # Get training data
  x_train=x[training_set,:,:]
  y_train=y[training_set,:]
  
  # Get testing data
  x_test=x[testing_set,:,:]
  y_test=y[testing_set,:]

  # Get validation data
  x_valid=x[valid_set,:,:]
  y_valid=y[valid_set,:]

  # Z-score inputs
  x_train_mean=np.nanmean(x_train,axis=0) 
  x_train_std=np.nanstd(x_train,axis=0) 
  x_train=(x_train-x_train_mean)/x_train_std 
  x_test=(x_test-x_train_mean)/x_train_std
  x_valid=(x_valid-x_train_mean)/x_train_std 

  # Zero-center outputs
  y_train_mean=np.nanmean(y_train,axis=0)
  y_train=y_train-y_train_mean
  y_test=y_test-y_train_mean
  y_valid=y_valid-y_train_mean

  ##### DECODING ######

  # Validation settings
  callbacks_list = [
  # Early stopping
  keras.callbacks.EarlyStopping(
      monitor='accuracy', 
      patience=1 
  ),
  # Save model
  keras.callbacks.ModelCheckpoint(
    filepath = '/content/drive/MyDrive/Spikes/trans/my_model.h5', 
    monitor='val_loss',
    save_best_only=True
  )]
  
  # Get hyperparameters using Bayesian optimization based on validation set R2 values
  # Define a function that returns the metric we are trying to optimize (R2 value of the validation set)
  # as a function of the hyperparameter we are fitting        
  # def evaluate(frac_dropout,n_epochs, n_heads):
  #     # n_layers=int(n_layers)
  #     frac_dropout=float(frac_dropout)
  #     n_epochs=int(n_epochs)
  #     n_heads=int(n_heads)
  #     model=transformer.CNNTransDecoder(chans=32,dropout=frac_dropout,num_epochs=n_epochs,num_heads=n_heads,verbose=1)
  #     model.fit(x_train,y_train,x_valid,y_valid,callbacks_list)
  #     y_valid_predicted=model.predict(x_valid)
  #     return np.mean(get_R2(y_valid,y_valid_predicted))
  
  # Do bayesian optimization
  # BO = BayesianOptimization(evaluate, {'frac_dropout': (0,.5), 'n_epochs': (2,8), 'n_heads': (2,8)})
  # BO.maximize(init_points=20, n_iter=1, kappa=10)
  # best_params=BO.res['max']['max_params']
  # frac_dropout=float(best_params['frac_dropout'])
  # n_epochs=np.int(best_params['n_epochs'])
  # # n_layers=np.int(best_params['n_layers'])
  # n_heads=np.int(best_params['n_heads'])

  frac_dropout=0.1
  n_epochs=5
  n_layers=1
  n_heads=8

  # Run model with above hyperparameters
  # model=transformer.TransformerDecoder(num_layers=n_layers,dropout=frac_dropout,num_epochs=n_epochs,num_heads=n_heads,verbose=1,mode='SpatialTemporal')
  # model=transformer.AttenGRUDecoder(num_units=400,dropout=frac_dropout,num_epochs=n_epochs,num_heads=n_heads,verbose=1)
  model=transformer.CNNTransDecoder(chans=32,dropout=frac_dropout,num_epochs=n_epochs,num_heads=n_heads,verbose=1,mode='STT')
  # model = GRUDecoder(units=400, dropout=0, num_epochs=5, verbose=1)
  # model = LSTMDecoder(units=400, dropout=0, num_epochs=5, verbose=1)
  # model = CNNDecoder(dropout=frac_dropout, num_epochs=n_epochs, verbose=1)
  # model = transformer.TransGRUDecoder(num_units=400,dropout=frac_dropout,num_epochs=n_epochs,num_heads=n_heads,verbose=1)
  
  model.fit(x_train,y_train,x_valid,y_valid,callbacks_list)
  y_test_predicted=model.predict(x_test)
  mean_r2[i]=np.mean(get_R2(y_test,y_test_predicted))   
  # Print test set R2 values
  R2s=get_R2(y_test,y_test_predicted)
  print('R2s:', R2s)
  # Add predictions of training/validation/testing to lists (for saving)            
  y_pred_all.append(y_test_predicted)
  y_train_pred_all.append(model.predict(x_train))
  y_valid_pred_all.append(model.predict(x_valid)) 

print('mean_r2:', mean_r2)
print ("\n")   
time_elapsed=time.time()-t1 # How much time has passed 

save_folder = '/content/drive/MyDrive/Spikes/trans'
# Save results
with open(save_folder+type+'_results.pickle','wb') as f:
  pickle.dump([mean_r2,y_pred_all,y_train_pred_all,y_valid_pred_all],f)
# Save ground truth results
with open(save_folder+type+'_ground_truth.pickle','wb') as f:
  pickle.dump([y_test_all,y_train_all,y_valid_all],f)
# Print time
print("time_elapsed:",time_elapsed)