# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:55:04 2020

@author: Achraf
"""

import numpy as np 

class NeuralNetwork:
    def __init__(self, inputLayerNeuronsNumber, hiddenLayerNeuronsNumber,outputLayerNeuronsNumber, learning_rate ):
        self.inputLayerNeuronsNumber = inputLayerNeuronsNumber
        self.hiddenLayerNeuronsNumber = hiddenLayerNeuronsNumber
        self.outputLayerNeuronsNumber = outputLayerNeuronsNumber
        self.learning_rate = learning_rate
            #weights and bias initialization
        self.hidden_weights=np.random.uniform(0,1,size=(self.inputLayerNeuronsNumber,self.hiddenLayerNeuronsNumber))            
        self.hidden_bias =np.random.uniform(0,1,size=(1,self.hiddenLayerNeuronsNumber))        
        self.output_weights =np.random.uniform(0,1,size=(self.hiddenLayerNeuronsNumber,self.outputLayerNeuronsNumber))        
        self.output_bias =np.random.uniform(0,1,size=(1,self.outputLayerNeuronsNumber))
        self.predicted_output=0;        
        
    def sigmoid (self,x):
        return 1/(1 + np.exp(-x)) 
    def sigmoid_derivative(self,x):
        return x * (1 - x) 
    def loss(self,x,y):
        return 0.5*pow((x-y),2)   
    
    def train(self,inputs,desired_output):
        print("*************************************train******************************")
        hidden_layer_in = np.dot(inputs,self.hidden_weights) + self.hidden_bias # somme(wi*xi)
        hidden_layer_out = self.sigmoid(hidden_layer_in) # la fonction d'activation f(somme)
        output_layer_in = np.dot(hidden_layer_out,self.output_weights) + self.output_bias # on calcule output de la deuxieme 
        self.predicted_output = self.sigmoid(output_layer_in) # on calcule la valeur de predicted 
        #Backward pass 
        error = desired_output - self.predicted_output # error
        d_predicted_output = error * self.sigmoid_derivative(self.predicted_output)  # error * f'()
        error_hidden_layer = d_predicted_output.dot(self.output_weights.T) # error*f'()* weight(ouptut)  "matrice" 
        d_hidden_layer = error_hidden_layer * self.sigmoid_derivative(hidden_layer_out)  # error_hiden_layer * f'(hidden(out))
        #Updating Weights and Biases (ajuster les weights)
        self.output_weights += hidden_layer_out.T.dot(d_predicted_output) * self.learning_rate # transposé(Hidden) * error * f'()* LR
        self.output_bias += np.sum(d_predicted_output,axis=0,keepdims=True) * self.learning_rate # somme du matrice(pred(out)) * LR 
        self.hidden_weights += inputs.T.dot(d_hidden_layer) * self.learning_rate # transposé(input)*d_hidden_layer
        self.hidden_bias += np.sum(d_hidden_layer,axis=0,keepdims=True) * self.learning_rate # somme du matrice(d_hidden_layer) * LR 
        print("updated hidden weights: ") 
        print(self.hidden_weights) 
        print("updated hi2dden biases: ") 
        print(self.hidden_bias) 
        print("updated output weights: ") 
        print(self.output_weights) 
        print("updated output biases: ") 
        print(self.output_bias) 
        
    def predict(self,desired_output):
        print("*************************************predict******************************")        
        print("Predicted out put: ",self.predicted_output) 
        loss_=self.loss(self.predicted_output,desired_output) 
        print("loss:",loss_) 

#intialisatiion des inputs ******************************************************************
print("Veuillez entrer learning_rate  : ")
learning_rate1 = input()
learning_rate=float(learning_rate1)
#Programme**********************************************************************************
nn=NeuralNetwork(2,2,1,learning_rate) 
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
desired_output = np.array([[0],[0],[0],[1]])
print("Veuillez entrer le nombre des iterations  : ")
iteration = input()
iterationC = int(iteration)
for i in range(iterationC):
    print("****************************iteration =",i+1,"**********************************")         
    nn.train(inputs,desired_output) 
    prediction=nn.predict(desired_output)
