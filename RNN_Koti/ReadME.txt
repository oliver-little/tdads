usage for Recurrent Neural Network:
python Recurrent-Neural-Network.py

Parameters that must be tweaked:
	test_images- x*28*28 numpy multidimensional array with entries between values inc. 0 and 255. In other words: x images that are sized 28x28 (provided in code x = 10,000 but x can be anything)
	test_labels- an x sized numpy single dimensional array that contains integer digits between values inc. 0 and 9. Please keep x consistent between test_labels and test_images, there is no validation in the code.

Outputs:
	Total accuracy, the loss, the total time it took to train, the total time it took to predict
