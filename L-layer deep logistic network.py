import numpy as np
def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def initialize_parameters_deep(layersdim):
    parameters = {}
    L = len(layersdim)
    for l in range(1,L):
        parameters['w'+str(l)] = np.random.randn(layersdim[l],layersdim[l-1])*0.01
        parameters['b'+str(l)] = np.zeros((layersdim[l],1))
    return parameters


def L_model_forward(X,parameters):
    
    L = len(parameters)//2
    
    cache = {}
    A = X    
    cache['a'+str(0)] = A
    for l in range(1,L):
        w = parameters['w' + str(l)]
        b = parameters['b' + str(l)]
        Z = np.dot(w,A) + b
        cache['z'+str(l)] = Z
        A = relu(Z)
        cache['a'+str(l)] = A
    w = parameters['w'+str(L)]
    b = parameters['b'+str(L)]
    Z = np.dot(w,A)+b
    A = sigmoid(Z)
    cache['a'+str(L)] = A
    cache['z'+str(L)] = Z

    return A,cache 

def L_model_backward(Y, parameters, cache):
    grads = {}
    L = len(parameters) // 2  # Number of layers in the neural network
    AL = cache['a' + str(L)]  # The output of the final layer
    Y = Y.reshape(AL.shape)  # Reshape Y to match the output of the last layer
    m = Y.shape[1]

    # Sigmoid backpropagation (output layer)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    dzL = dAL * sigmoid_derivative(AL)
    grads["dw" + str(L)] = (1 / m) * np.dot(dzL, cache["a" + str(L-1)].T)  # Gradients of W for the last layer
    grads["db" + str(L)] = (1 / m) * np.sum(dzL, axis=1, keepdims=True)

    # Backpropagate for the hidden layers (1 to L-1)
    for l in reversed(range(1, L)):  # Start from L-1 down to 1
        dz = np.dot(parameters["w" + str(l + 1)].T, dzL) * relu_derivative(cache["z" + str(l)])
        grads["dw" + str(l)] = (1 / m) * np.dot(dz, cache["a" + str(l - 1)].T)  # Gradients of W for hidden layers
        grads["db" + str(l)] = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        dzL = dz

    return grads
def compute_cost(Y,AL):
    
    m = Y.shape[1]
    cost = - (1 / m) * np.sum(Y * np.log(AL) + (1 - Y) * np.log(1 - AL))
    return cost

def update_parameters(grads,parameters,alpha):

    L = len(parameters)//2
    for i in range(1,L+1):

        w= parameters['w'+str(i)]
        b= parameters['b'+str(i)]
        w = w - alpha*grads['dw'+str(i)]
        b = b - alpha*grads['db'+str(i)]
        parameters['w'+str(i)] = w
        parameters['b'+str(i)] = b

    return parameters
def predict(X, parameters, Y=None):
    
    AL, _ = L_model_forward(X, parameters)
    
    predictions = (AL > 0.5).astype(int)
    
    if Y is not None:
    
        accuracy = np.mean(predictions == Y)
        return predictions, accuracy
    
    return predictions

def L_model(layersdim,X,Y,alpha,iterations):
    parameters = initialize_parameters_deep(layersdim)
    costs= []
    for i in range(iterations):

        
        
        A,cache = L_model_forward(X,parameters)
        grads = L_model_backward(Y,parameters,cache)
        parameters = update_parameters(grads,parameters,alpha)
        if i%100 == 0 :
            c =compute_cost(Y,A)
            print('cost after ',i,'iter: ',c)
            costs.append(c)
    
    return parameters,costs




from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load the dataset
data = load_breast_cancer()
X = data.data  # Features
Y = data.target  # Labels (0 or 1)

# Preprocess the data
scaler = StandardScaler()  # Standardize features (zero mean, unit variance)
X = scaler.fit_transform(X)  # Scale features
Y = Y.reshape(1, -1)         # Reshape Y to (1, m)
X = X.T                      # Transpose X to (n_x, m)
X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.2, random_state=42)
X_train, X_test = X_train.T, X_test.T  # Transpose back to (n_x, m)
Y_train, Y_test = Y_train.T, Y_test.T  # Transpose back to (1, m)


# Print shapes for verification
print("Shape of X:", X.shape)  # (n_x, m)
print("Shape of Y:", Y.shape)  # (1, m)

layers =[X_train.shape[0],30,12,1]

param ,costlist= L_model(layers,X_train,Y_train,0.1,2500)

train, accuracytrain =predict(X_train,param,Y_train)
test, accuracytest = predict(X_test,param,Y_test)

print(f"train set accuracy {accuracytrain}")
print(f"test set accuracy {accuracytest}")

