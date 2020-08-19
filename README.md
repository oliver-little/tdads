# TDADS
### Technology Degree Apprentice Data Science Project
This is a fork of [this repository](https://github.com/georgeahill/tdads) in its final state for archival purposes.

I worked on this project from June 2020 to August 2020 with other PwC Technology Degree Apprentices, and the project was split into two tasks.

## Task 1 *(June-July)* 
#### Categorisation of digits using the MNIST dataset:
  * This was a more defined task, with the goal simply being to create the most accurate model for categorising digits.
  * We created various models - both neural networks and non-neural networks - and produced a [presentation](https://docs.google.com/presentation/d/1RmAVyIR17ulPZgrTaQsucfRf7IYmCaQbeySqlb79uaw/) detailing our findings.

Personally, I worked on three models:
  * Support Vector Machine (SVM): Implemented using sklearn
  * Naive Bayes: Custom implementation using numpy
  * Random Forest: Implemented using sklearn
  
## Task 2 *(July-August)*
#### Sentiment analysis of tweets directed at US Airlines:
  * This task was much more loosely defined, with the main goal being simply to produce a classifier to predict the sentiment of a tweet directed at an airline (positive, negative or neutral).
  * To solve this problem, we created a few baseline models, and a final Recurrent Neural Network model.
  * This task's other requirement was to get insights from the dataset which could be then used to make recommendations to businesses.
  * This involved a significant amount of exploratory analysis in order to gather these insights into the dataset.
  * We then produced another [presentation](https://docs.google.com/presentation/d/1XHX1na4_FYFszKSejv5C_ssDy6Ls2ebE3vI-ee0Y2Uw) detailing our findings.
  
I worked on a few things in this task:
  * Extracting locations from tweet texts using a keyword-based system, and reference lists of airport codes and city names.
  * Latent Semantic Analysis (automatic categorisation of data) in order to find the topics that appear most in the dataset.
  * A baseline Naive Bayes classifier, implemented using sklearn.
