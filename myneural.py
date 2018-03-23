from numpy import exp, array, random, dot
class NeuralNetowrk():
	def __init__(self):		
		self.weights = random.random((5, 1))
	def __sigmoid_der(self, x):
		return x * (1 - x)
	def __sigmoid(self, x):
		return 1 / (1 + exp(-x))
	def train(self, inputs,outputs,number_of_iterations):
		for iteration in range(number_of_iterations):
			output = self.anwsear(inputs)
			globalerror = outputs - output
			adjust = dot(inputs.T,globalerror * self.__sigmoid_der(output))			
			self.weights += adjust
	def anwsear(self,inputs):
		return self.__sigmoid(dot(inputs, self.weights))
if __name__ == "__main__":
	neural_network = NeuralNetowrk()
	training_inputs = array([[1, 0, 1,0,1], [1, 1, 1,1,1], [1, 0, 1,0,0], [0, 1, 1,1,1]])
	training_outputs = array([[0, 1, 0, 1]]).T
	neural_network.train(training_inputs, training_outputs, 5000)
	print (neural_network.anwsear(array([1, 0, 0,0,1])))
