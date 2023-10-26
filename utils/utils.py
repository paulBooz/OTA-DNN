import numpy as np
import csv
import matplotlib.pyplot as plt

def generate_data_SF(n_users, n_samples, output_nodes):
    
    #uniform distribution with zero mean unit variance
    samples = np.random.uniform(-np.sqrt(3), np.sqrt(3), (n_samples, n_users))
    
    #Rayleigh channels
    h = np.random.rayleigh(1/np.sqrt(2), (n_samples, n_users, output_nodes))

    #create X_train with the form [x1, h1, x2, h2,...., xN, hN], with the entries being columns
    X = np.zeros((n_samples, n_users + n_users*output_nodes))
    X[:, ::(output_nodes+1)] = samples
    for i in range(n_users):
        for j in range(output_nodes):
            X[:, i*(output_nodes+1)+1+j] = h[:, i, j]

    #y = (1/m)(x1 + x2 +. .. +xN + x1*x2*...*xN)^2
    m = (n_users*np.sqrt(3) + np.sqrt(3)**(n_users))**2
    
    sum_ = np.zeros(np.shape(X[:,0]))
    prod_ = np.ones(np.shape(X[:,0]))
    for i in range(n_users):
        sum_ = sum_ + X[:, i*(output_nodes+1)]
        prod_ = prod_*X[:, i*(output_nodes+1)]
    
    Y = (1/m)*(sum_ + prod_)**2
    
    return X, Y

def generate_data_FF(n_users, n_samples):
    
    #uniform distribution with zero mean unit variance
    samples = np.random.uniform(-np.sqrt(3), np.sqrt(3), (n_samples, n_users))
    
    #Rayleigh channels
    h = np.random.rayleigh(1/np.sqrt(2), (n_samples, n_users))
    #h = np.ones((n_samples, n_users))

    #create X_train with the form [x1, h1, x2, h2,...., xN, hN], with the entries being columns
    X = np.zeros((n_samples, 2*n_users))
    X[:, ::2] = samples
    X[:, 1::2] = h

    #y = (1/m)(x1 + x2 +. .. +xN + x1*x2*...*xN)^2
    m = (n_users*np.sqrt(3) + np.sqrt(3)**(n_users))**2
    
    sum_ = np.zeros(np.shape(X[:,0]))
    prod_ = np.ones(np.shape(X[:,0]))
    for i in range(n_users):
        sum_ = sum_ + X[:, 2*i]
        prod_ = prod_*X[:, 2*i]
    
    Y = (1/m)*(sum_ + prod_)**2

    return X, Y

def save_as_csv(data, path):
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(data)

def make_plot_SNR(cDNN, DOTAE, oracle_benchmark, SNR):
    plt.plot(SNR, cDNN, label='cDNN')
    plt.plot(SNR, DOTAE, label='DOTAE')
    plt.plot(SNR, oracle_benchmark, label='oracle_benchmark')

    plt.title('Average MSE vs recieved SNR')
    plt.xlabel('recieved SNR (dB)')
    plt.ylabel('Average MSE')

    plt.yscale('log')
    plt.legend()
    plt.show()
    
def make_plot_users(cDNN, DOTAE, oracle_benchmark, number_of_users):
    plt.plot(number_of_users, cDNN, label='cDNN')
    plt.plot(number_of_users, DOTAE, label='DOTAE')
    plt.plot(number_of_users, oracle_benchmark, label='oracle_benchmark')

    plt.title('number_of_users')
    plt.xlabel('number_of_users')
    plt.ylabel('Average MSE')

    plt.yscale('log')
    plt.legend()
    plt.show()
