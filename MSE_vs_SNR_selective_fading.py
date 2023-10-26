import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from sklearn.metrics import mean_squared_error
from tensorflow.keras.callbacks import ReduceLROnPlateau
from utils.utils import generate_data_SF, save_as_csv, make_plot_SNR

av_received_SNR = np.linspace(25, 30, 2)
std_noise_vector = 10**(-av_received_SNR/20)
n_users = 2

#initializations
test_loss_centralized = []
test_loss_DOTAE = []
test_loss_oracle_benchmark = []

#number of output nodes of the local DNNs or size of the augmented vector
output_nodes = 2

def custom_tanh(x):
    return np.sqrt(1/output_nodes)*tf.nn.tanh(x)

#this loss satisfies the power constraint
def power_constraint_activation(x):
    sum_of_squares = tf.reduce_sum(tf.square(x), axis=-1, keepdims=True)
    x_normalized = (x*np.sqrt(1/output_nodes))/tf.sqrt(sum_of_squares)
    return x_normalized

#iterate through the noise vector
for iters in range(len(std_noise_vector)):
    std_noise = std_noise_vector[iters]
    print()
    print('SNR:', av_received_SNR[iters])
    
    #generate training dataset
    train_samples = 50000
    X_train, Y_train = generate_data_SF(n_users, train_samples, output_nodes)
    
    #generate testing dataset
    test_samples = 30000
    X_test, Y_test = generate_data_SF(n_users, test_samples, output_nodes)
    
    #-----------------------------------------------------
    #------------------ Create the layers ----------------
    
    initializer = tf.keras.initializers.GlorotUniform()
    input_signals = []
    input_channels = []
    inputs = []
    for i in range(n_users):
        #create input signal
        inp_signal = tf.keras.Input(shape=(1,))
        input_signals.append(inp_signal)
        
        #create input channel
        inp_channel = tf.keras.Input(shape=(output_nodes,))
        input_channels.append(inp_channel)
        
        #concatenate input signals and channels
        concatenated = tf.keras.layers.concatenate([inp_signal, inp_channel])
        inputs.append(concatenated)
        
    f = []
    local_models = []
    for i in range(n_users):
        f.append(tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer)(inputs[i]))
        f[i] = tf.keras.layers.Dense(256, activation='relu', kernel_initializer=initializer)(f[i])
        f[i] = tf.keras.layers.Dense(output_nodes, activation=power_constraint_activation, kernel_initializer=initializer)(f[i])
        
        #for the local model, there is no multiplication with channel. Take the layer as is.
        local_model = tf.keras.Model(inputs=[input_signals[i], input_channels[i]], outputs=f[i])
        local_models.append(local_model)
        
        #for the cDNN, multiply the output with the channel
        f[i] = Lambda(lambda x: x[0]*x[1])([f[i], input_channels[i]])
    
    #merge the layers by addition 
    merge = tf.keras.layers.Add()(f)
    
    #add gaussian noise
    noise = tf.random.normal(shape=tf.shape(merge), mean=0, stddev=std_noise/np.sqrt(output_nodes))
    merge = tf.keras.layers.Add()([merge, noise])
    
    #define the final layers
    g = tf.keras.layers.Dense(256, activation='relu')(merge)
    g = tf.keras.layers.Dense(1)(g)
    
    #model of the BS (post-processing function in the DOTAE)
    BS_model = tf.keras.Model(inputs=merge, outputs=g)
    
    #--------------------------------------------------------------------
    #---------------- Build and train the cDNN --------------------------
        
    #build model (this is the cDNN)
    inputs_centralized = []
    for i in range(n_users):
        inputs_centralized.append(input_signals[i])
        inputs_centralized.append(input_channels[i])
    model = tf.keras.Model(inputs=[inputs_centralized[i] for i in range(2*n_users)], outputs=g)
    
    #compile the cDNN
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.99, beta_2=0.999, epsilon=1e-08),
                  loss='mean_squared_error')
    
    #transform X_train for feeding the DNNs
    X_train_list = []
    for i in range(n_users):
        X_train_list.append(X_train[:, i*(output_nodes+1)])
        X_train_list.append(X_train[:, i*(output_nodes+1)+1:i*(output_nodes+1)+1+output_nodes])
    
    #transform X_test for feeding the DNNs
    X_test_list = []
    for i in range(n_users):
        X_test_list.append(X_test[:, i*(output_nodes+1)])
        X_test_list.append(X_test[:, i*(output_nodes+1)+1:i*(output_nodes+1)+1+output_nodes])
        
    #train the cDNN
    model.fit(X_train_list, Y_train, epochs=1, batch_size=64)
    
    #save the weights of the trained cDNN
    weights = model.get_weights()
    
    #evaluate the cDNN
    loss = model.evaluate(X_test_list, Y_test)
    print('Test loss:', loss)
    
    pred_centralized = model.predict(X_test_list)
    
    #-----------------------------------------------------------------------
    #--------------------Implementaton of DOTAE-----------------------------
    
    #copy the weights of the cDNN to each local model (pre-processing functions in DOTAE)
    for i in range(n_users):
        w = local_models[i].get_weights()
        w[0] = weights[2*i]
        w[1] = weights[2*i+1]
        w[2] = weights[2*(n_users+i)]
        w[3] = weights[2*(n_users+i)+1]
        w[4] = weights[4*n_users+2*i]
        w[5] = weights[4*n_users+2*i+1]
        local_models[i].set_weights(w)
    
    #copy the weights of the cDNN to the BS model (post-processing function in DOTAE)
    w_BS = BS_model.get_weights()
    for i in range(len(w_BS)):
        w_BS[i] = weights[6*n_users+i]
    BS_model.set_weights(w_BS)
    
    #channel multiplication and summation over-the-air in DOTAE
    sum_ = np.zeros((test_samples, 1))
    local_predictions = []
    for i in range(n_users):
        local_predictions.append(local_models[i].predict([X_test_list[2*i], X_test_list[2*i+1]]))
        sum_ = sum_ + X_test_list[2*i+1]*local_predictions[i]
    
    #std_noise = np.sqrt(0.01)
    sum_ = sum_ + np.random.normal(0, std_noise/np.sqrt(output_nodes), np.shape(sum_))
    
    #final prediction/function approximation of the BS (output of DOTAE)
    pred = BS_model.predict(sum_)
    
    #-----------------------------------------------------------
    #------------------- Oracle benchmark ----------------------
    
    #normalize the inputs
    Y_test_norm = (Y_test - np.mean(Y_test))/np.std(Y_test)
    
    #estimation for the oracle benchmark
    h_s = np.random.rayleigh(1/np.sqrt(2), (test_samples, 1))
    #h = np.ones((test_samples, 1))
    Y_recieved = np.sqrt(n_users)*h_s[:,0]*Y_test_norm + np.random.normal(0, std_noise, np.shape(Y_test))
    Y_recieved = Y_recieved/((std_noise**2+n_users*h_s[:,0]**2)/(np.sqrt(n_users)*h_s[:,0]))
    Y_hat = np.std(Y_test)*Y_recieved + np.mean(Y_test)

    #-------------------------------------------------
    
    #cDNN loss (this is equal to loss. Just checking.)
    test_loss_centralized.append(mean_squared_error(Y_test, pred_centralized))
    
    #loss of DOTAE - proposed implementation
    test_loss_DOTAE.append(mean_squared_error(Y_test, pred))
    
    #loss of the oracle benchmark
    test_loss_oracle_benchmark.append(mean_squared_error(Y_test, Y_hat))
    

#plot
make_plot_SNR(test_loss_centralized, test_loss_DOTAE, test_loss_oracle_benchmark, av_received_SNR)

#save data
save_as_csv(test_loss_centralized, 'simulation_results/test_loss_centralized.csv')
save_as_csv(test_loss_DOTAE, 'simulation_results/test_loss_DOTAE.csv')
save_as_csv(test_loss_oracle_benchmark, 'simulation_results/test_oracle_benchmark.csv')


