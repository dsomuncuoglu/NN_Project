import numpy as np

input_=np.array([1.88,1.56])
weights_=np.array([1.45,-0.66])
bias_=np.array([0.0])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def create_prediction(input,weights,bias):
    layer_1=np.dot(input_,weights_)+bias_
    layer_2=sigmoid(layer_1)
    return layer_2

def sigmoid_deriv(x):
    return sigmoid(x)-(1-sigmoid(x))

prediction_=create_prediction(input=input_,weights=weights_,bias=bias_)

print(f"The results : {prediction_}")

input_[0]=input_[0]+0.25
input_[1]=input_[1]+0.35

prediction_=create_prediction(input=input_,weights=weights_,bias=bias_)

target_=0

mse=np.square(prediction_-target_)
print(f"Prediction : {prediction_} Error : {mse}")


derivative_=2*(prediction_-target_)
print(f"Derivative : {derivative_}")

weights_=(weights_-derivative_)
prediction_=create_prediction(input=input_,weights=weights_,bias=bias_)
error_=(prediction_-target_)**2

print(f"Last Prediction : {prediction_} Error : {error_}")

derror_dpredicition=2*(prediction_-target_)
layer_1=np.dot(input_,weights_)+bias_
dprediction_dlayer1=sigmoid_deriv(layer_1)
dlayer1_dbias=1

derror_dbias=(derror_dpredicition*dprediction_dlayer1*dlayer1_dbias)