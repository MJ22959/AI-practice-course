import numpy as np
from preprocess_Data import features, targets, features_test, targets_test
import random

X = features.values
y = targets.values

np.random.seed(42)

def sigmoid(x):
    """Calculate sigmoid"""
    return 1 / (1 + np.exp(-x))


# hyperparameters
n_hidden = 3  # number of hidden units
epochs = 5000
learning_rate = 0.5

n_records, n_features = features.shape

last_loss = None
# initialize weights
weights_input_hidden = np.random.normal(scale=1 / n_features ** -.5,
                                        size=(n_features, n_hidden))
weights_hidden_output = np.random.normal(scale=1 / n_features ** -.5,
                                         size=n_hidden)
                                     
for e in range(epochs):

    del_w_input_hidden = np.zeros(weights_input_hidden.shape)
    del_w_hidden_output = np.zeros(weights_hidden_output.shape)
    
    # forward pass
    # calculate the output
    hidden_input = np.dot(X, weights_input_hidden)
    hidden_activations = sigmoid(hidden_input)
    output_layer_input = np.dot(weights_hidden_output, hidden_activations.T)
    output = sigmoid(output_layer_input)
    
    # backward pass
    # calculate the error     
    error = y - output    

    # calculate the error gradient in output unit
    temp_o = np.multiply(output,(1 - output))
    output_error = np.multiply(error,temp_o).reshape(-1,1)
        
    # propogate error to hidden layer
    tmp_ho = np.dot(output_error, weights_hidden_output.reshape(-1,1).T)
    sig_prod = np.multiply(hidden_activations, (1 - hidden_activations)) 
    hidden_error = np.multiply(tmp_ho,sig_prod)
    
        
    # update the change in weights
    del_w_hidden_output = np.multiply(output_error , hidden_activations)
    del_w_hidden_output = np.sum(del_w_hidden_output, axis =0)
    
    del_w_input_hidden = np.dot(hidden_error.T , X).T
     
    # update weights
    weights_hidden_output += learning_rate * del_w_hidden_output * (1/n_records)
    weights_input_hidden += learning_rate * del_w_input_hidden * (1/n_records)

    # printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        #hidden_activations = sigmoid(np.dot(X[random.randint(0, int(len(features)))], weights_input_hidden))
        hidden_activations = sigmoid(np.dot(X[int(len(features)-1)], weights_input_hidden))
        out = sigmoid(np.dot(hidden_activations,
                             weights_hidden_output))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss              
        

# calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, weights_input_hidden))
out = sigmoid(np.dot(hidden, weights_hidden_output))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))                      

