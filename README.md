# TDADS
TODO: description


## Requirements

## Usage

## K-Nearest Neighbours

usage for K-Nearest_Neighbours without parallel processing:
python K-Nearest-Neighbours.py

usage for K-Nearest_Neighbours with parallel processing:
python K-Nearest-Neighbours_Parallel.py

Parameters that must be tweaked:
	*test_images: x*28*28 numpy multidimensional array with entries between values inc. 0 and 255. In other words, x images that are sized 28x28 (provided in code x = 10,000 but x can be anything)
	*test_labels: an x sized numpy single dimensional array that contains integer digits between values inc. 0 and 9. Please keep x consistent between test_labels and test_images, there is no validation in the code.
	*(for Parallel processing only)
	THREADS_N- the number of concurrent threads you want to run on your machine (provided in code is 16)

Hyperparameters that you may want to tweak:
	-K_VALUE- the value of K in K Nearest Neighbours (default is 10 and seems to be fine with the test set)

Outputs after every THREADS_N images predicted:
	-Number of images predicted, current accuracy, estimated time for completion from now

Outputs after all images predicted:
	-Number of images predicted, total accuracy, the total time it took for completion


Note: When running these scripts, you may see warnings or errors but just ignore them since the code still runs in the background and there are prints to update its progression (approx. every 1-2 minutes if THREADS_N = 16).


## Recurrent Neural Network

usage for Recurrent Neural Network:
python Recurrent-Neural-Network.py

Parameters that must be tweaked:
	-test_images: x*28*28 numpy multidimensional array with entries between values inc. 0 and 255. In other words, x images that are sized 28x28 (provided in code x = 10,000 but x can be anything)
	-test_labels: an x sized numpy single dimensional array that contains integer digits between values inc. 0 and 9. Please keep x consistent between test_labels and test_images, there is no validation in the code.

Outputs:
	-Total accuracy, the loss, the total time it took to train, the total time it took to predict
