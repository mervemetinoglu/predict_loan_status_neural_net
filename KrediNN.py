import numpy as np

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
  #  f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

def mse_loss(y_true, y_pred):
  # Mean Square Error Loss function 
  return ((y_true - y_pred) ** 2).mean()
  
class KrediNN:
    '''
      input
        o     hidden
        o       o     output
        o               o   
        o       o
        o
        o
        o
    '''
    
    def __init__(self, input_dim):
        self.input_dim = input_dim
        
        # for 2 norons -> input_dim * 2
        # for 1 output noron -> 2
        self.weights = np.random.rand(2 * input_dim + 2) 
        
        # bias values
        self.bias = [np.random.normal(), np.random.normal(), np.random.normal()]
    
    def test(self, x):
        input_dim = self.input_dim
        weights = self.weights
        bias = self.bias

        # [0:input_dim] -> first layer's weights
        h1 = sigmoid(np.dot(weights[:input_dim], x) + bias[0])

        # [input_dim:input_dim*2] -> second layer's weights
        h2 = sigmoid(np.dot(weights[input_dim:2*input_dim], x) + bias[1])

        # [2*input_dim:] -> output layer's weights
        o1 = sigmoid(np.dot(weights[2*input_dim:], np.array([h1, h2])) + bias[2])
        return o1
    
    def train(self, x_train, y_train):
        learning_rate = 0.1
        epochs = 500

        input_dim = self.input_dim

        for epoch in range(epochs):
            for x, y_true in zip(x_train, y_train):
                
                # (x1​*w1​) + (x2*w2​) +...+ (xn*wn) + b
                sum_h1 = np.dot(self.weights[:input_dim], x) + self.bias[0]
                h1 = sigmoid(sum_h1)

                sum_h2 = np.dot(self.weights[input_dim:2*input_dim], x) + self.bias[1]
                h2 = sigmoid(sum_h2)

                # Output
                sum_o1 = np.dot(self.weights[2*input_dim:], [h1, h2]) + self.bias[2]
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # --- back propagation ---
                
        
                d_L_d_ypred = -2 * (y_true - y_pred)

 
                d_ypred_d_wo1 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_wo2 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.weights[2*input_dim:2*input_dim+1] * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.weights[2*input_dim+1:] * deriv_sigmoid(sum_o1)

     
                d_h1_d_w_input_dim = []
                for i in range(input_dim):
                    d_h1_d_w_input_dim.append(x[i] * deriv_sigmoid(sum_h1))

                d_h1_d_b1 = deriv_sigmoid(sum_h1)


                d_h2_d_w_input_dim = []
                for i in range(input_dim):
                    d_h2_d_w_input_dim.append(x[i] * deriv_sigmoid(sum_h2))
                    
                d_h2_d_b2 = deriv_sigmoid(sum_h2)


                for i in range(input_dim):
                    self.weights[i] -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w_input_dim[i]
                
                self.bias[0] -= learning_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1


                for i in range(input_dim):
                    self.weights[input_dim + i] -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w_input_dim[i]
            
                self.bias[1] -= learning_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

      
                self.weights[2*input_dim:2*input_dim+1] -= learning_rate * d_L_d_ypred * d_ypred_d_wo1
                self.weights[2*input_dim+1:] -= learning_rate * d_L_d_ypred * d_ypred_d_wo2
                self.bias[2] -= learning_rate * d_L_d_ypred * d_ypred_d_b3
                
            # --- calculate loss value ---
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.test, 1, x_train)
                loss = mse_loss(y_train, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))