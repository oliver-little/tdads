The Davids_features_functions file contains two functions intended for outside use:

Predict() takes an array of images (I think assuming a numpy array) of the format produced by 
mnist.load_data, but I hope I've actually made it more general - certainly for size, but it might also take any 8 bit greyscale image array.

if you like compact, unreadable, code, you could condense most of it to:

results.append(Interpret(FindFeatures(images[tin])))

... but I wouldn't recommend it...

It returns a list of the predictions for each image in the input array.


Analyse() takes two lists: the "correct" results and the "predicted" results. It will generate a basic confusion table, with the rows corresponding to the "correct" values, and the columns to the predicted values, normalised to a percentage. Note that there are 11 columns (column 10 holds any digits that haven't been classified: at present that should always be 0!). It also calculates the percentage correctly identified.

Note that if you wanted to call Analyse() from inside Predict() you would need to amend Predict() to include the "correct" answers as a parameter, which would be perfectly possible, but not in line with your request.



There are some "internal" functions:

FindFeatures() takes a single image, converts it to B&W, and finds the features, partly using
Findloop(). It returns an array of 38 elements, which are the features detected: details are at the top of the function.

You might conceivably want to call FindFeatures to pass the values to some other model, but more likely you'd adapt Predict() to save the features in a larger array, so you can pass it to (e.g.) a neural net for training.

FindLoop() takes various parameters from FindFeatures(), and searches (part of) an image for a vertical or horizontal curved region, and whether or not that region is closed.

Interpret() takes an array of feature data from a single image, and attempts to identify what the digit was. It returns a single value, which is the "best guess". It returns 10 if it failed to make a guess.