import numpy as np

class L_model():

    def __init__(self,X,Y,layers):
        self.X = X
        self.Y = Y
        self.layers = layers
        self.parameters = self.initialize_parameters_deep(layers)
        self.grads = {}
        self.cost = []
    
    def relu(self,z):
        return np.maximum(0, z)

    def relu_derivative(self,z):
        return (z > 0).astype(float)

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self,a):
        return a * (1 - a)
    
    def L_model_forward(self,X):
        L = len(self.parameters)//2
        cache = {}
        A = X    
        cache['a'+str(0)] = A
        for l in range(1,L):
            w = self.parameters['w' + str(l)]
            b = self.parameters['b' + str(l)]
            Z = np.dot(w,A) + b
            cache['z'+str(l)] = Z
            A = self.relu(Z)
            cache['a'+str(l)] = A
        w = self.parameters['w'+str(L)]
        b = self.parameters['b'+str(L)]
        Z = np.dot(w,A)+b
        A = self.sigmoid(Z)
        cache['a'+str(L)] = A
        cache['z'+str(L)] = Z

        return A,cache 
    def initialize_parameters_deep(self,layersdim):
            parameters = {}
            L = len(layersdim)
            for l in range(1,L):
                parameters['w'+str(l)] = np.random.randn(layersdim[l],layersdim[l-1])*np.sqrt(2/layersdim[l-1])
                parameters['b'+str(l)] = np.zeros((layersdim[l],1))
            return parameters
    def L_model_backward(self,Y, parameters, cache):
        grads = {}
        L = len(parameters) // 2  
        AL = cache['a' + str(L)]  
        Y = Y.reshape(AL.shape)  
        m = Y.shape[1]
        epsi = 1e-8

        # Sigmoid backpropagation (output layer)
        dAL = - (np.divide(Y, AL+epsi) - np.divide(1 - Y, 1 - AL+epsi))
        dzL = dAL * self.sigmoid_derivative(AL)
        grads["dw" + str(L)] = (1 / m) * np.dot(dzL, cache["a" + str(L-1)].T)  # Gradients of W for the last layer
        grads["db" + str(L)] = (1 / m) * np.sum(dzL, axis=1, keepdims=True)

        # Backpropagate for the hidden layers (1 to L-1)
        for l in reversed(range(1, L)):  
            dz = np.dot(parameters["w" + str(l + 1)].T, dzL) * self.relu_derivative(cache["z" + str(l)])
            grads["dw" + str(l)] = (1 / m) * np.dot(dz, cache["a" + str(l - 1)].T)  # Gradients of W for hidden layers
            grads["db" + str(l)] = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            dzL = dz
        self.grads = grads
        return grads
    
    def compute_cost(self,Y,AL):
    
        m = Y.shape[1]
        epsi = 1e-8
        cost = - (1 / m) * np.sum(Y * np.log(AL+epsi)) + (1 - Y) * np.log(1 - AL+epsi)
        return cost
    
    def update_parameters(self,alpha):

        L = len(self.parameters)//2
        for i in range(1,L+1):

            w= self.parameters['w'+str(i)]
            b= self.parameters['b'+str(i)]
            w = w - alpha*self.grads['dw'+str(i)]
            b = b - alpha*self.grads['db'+str(i)]
            self.parameters['w'+str(i)] = w
            self.parameters['b'+str(i)] = b
        
        return self.parameters
    def predict(self,X, Y=None):
    
        AL, _ = self.L_model_forward(X)
    
        predictions = (AL > 0.5).astype(int)
    
        if Y is not None:
    
            accuracy = np.mean(predictions == Y)
            return predictions, accuracy
    
        return predictions
    def build(self,alpha,iterations):
        parameters = self.initialize_parameters_deep(self.layers)
        costs= []
        for i in range(iterations):

        
        
            A,cache = self.L_model_forward(self.X)
            self.L_model_backward(self.Y,parameters,cache)
            parameters = self.update_parameters(alpha)
            if i%100 == 0 :
                c =self.compute_cost(self.Y,A)
                print('cost after ',i,'iter: ',c)
                costs.append(c)
        self.parameters = parameters
        return parameters,costs





from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Load the dataset
data = load_breast_cancer()
X = data.data  # Features
Y = data.target  # Labels (0 or 1)

# Preprocess the data
scaler = StandardScaler()  
X = scaler.fit_transform(X)  
Y = Y.reshape(1, -1)         
X = X.T                      
X_train, X_test, Y_train, Y_test = train_test_split(X.T, Y.T, test_size=0.2, random_state=42)
X_train, X_test = X_train.T, X_test.T  
Y_train, Y_test = Y_train.T, Y_test.T  


# Print shapes for verification
print("Shape of X:", X.shape)  # (n_x, m)
print("Shape of Y:", Y.shape)  # (1, m)

layers =[X_train.shape[0],30,12,1]

logi= L_model(X_train,Y_train,layers)
parameters,costs = logi.build(iterations=2500,alpha=0.1)

train, accuracytrain =logi.predict(X_train,Y_train)
test, accuracytest = logi.predict(X_test,Y_test)

print(f"train set accuracy {accuracytrain}")
print(f"test set accuracy {accuracytest}")


