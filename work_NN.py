from train_NN import NeuralNetwork
from train_NN import np
import matplotlib.pyplot as plt
import time

tic=time.perf_counter()
input_=np.array([[4,1.5],[2,4.6],[4,1.5],[1.7,4],[3.5,4.5],[2,0.5],[5.5,2.3],[7.5,3.3]])

targets=np.array([0, 1, 0, 1, 0, 1, 1, 0])
learning_rate=0.5
neural_network=NeuralNetwork(learning_rate)
training_errors=neural_network.train(input_,targets,10000)
toc=time.perf_counter()
print(f"Timer : {toc-tic}")
plt.plot(training_errors)
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()




