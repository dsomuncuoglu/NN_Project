import numpy as np

class NeuralNetwork:
    def __init__(self,learning_rate):
        self.weights=np.array([np.random.randn(),np.random.randn()])
        self.bias=np.random.randn()
        self.learning_rate=learning_rate

    def sigmoid_(self,x):
        return 1/(1+np.exp(-x))

    def sigmoid_deriv_(self,x):
        return self.sigmoid_(x)*(1-self.sigmoid_(x))

    def prediction(self,input_):
        layer_1=np.dot(input_,self.weights)+self.bias
        layer_2=self.sigmoid_(layer_1)
        prediction1=layer_2
        return prediction1

    def compute_gradients_(self,input_,target_):
        layer_1=np.dot(input_,self.weights)+self.bias
        layer_2=self.sigmoid_(layer_1)
        prediction1=layer_2

        derror_dpredicition=2*(prediction1-target_)
        dprediction_dlayer1=self.sigmoid_deriv_(layer_1)
        dlayer1_dbias=1
        dlayer1_dweights=(0*self.weights)+(1*input_)

        derror_dbias=(derror_dpredicition*dprediction_dlayer1*dlayer1_dbias)
        derror_weights=(derror_dpredicition*dprediction_dlayer1*dlayer1_dweights)

        return derror_dbias,derror_weights

    def update_parameters_(self,derror_dbias,derror_dweights):
        self.bias=self.bias-(derror_dbias*self.learning_rate)
        self.weights=self.weights-(derror_dweights*self.learning_rate)

    def train(self,inputs_,targets,iterations):
        cumulative_errors=[]
        for current_iterations in range(iterations):
            random_data_index = np.random.randint(len(inputs_))

            input_=inputs_[random_data_index]
            target=targets[random_data_index]

            derror_dbias,derror_dweights=self.compute_gradients_(input_,target)

            self.update_parameters_(derror_dbias,derror_dweights)

            if current_iterations%100==0:
                cumulative_error=0
                for data_instance_index in range(len(inputs_)):
                    data_point=inputs_[data_instance_index]
                    target=targets[data_instance_index]

                    prediction=self.prediction(data_point)
                    error=np.square(prediction-target)

                    cumulative_error=cumulative_error+error
                cumulative_errors.append(cumulative_error)
        return cumulative_errors