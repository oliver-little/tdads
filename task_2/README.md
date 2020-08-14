# Location Recognition

## Usage

- LocationRecognition.py - run as main module, outputs to:
  _ code_output/tweets_with_locations.csv
  _ code_output/locations_totals.csv

## Requirements

- tweets.csv
- preproc.py
- world-cities.csv
- airport-codes.csv

# Naive Bayes

## Usage

- NaiveBayes.py - run as main module

## Requirements

- tweets.csv
- preproc.py

# Latent Semantic Analysis - LDA

## Usage

- LDA_gensim.py - run as main module, outputs to code_output/airline_topics/

## Requirements

- tweets.csv
- preproc.py

# Latent Semantic Analysis - SVD

## Usage

- LSA.py - run as main module, outputs to code_output/topic_tweets.csv

## Requirements

- preprocessed_negative_tweets.csv
- preproc.py

# Logistical Regressor

## Usage

```bash
python logistic_regressor.py
```

It will then ask for you to enter the name of the file of processed tweets you want to use.

The first run of the program is the slowest. It has to generate a tweet vectors file which takes about five minutes on my laptop.

Once it has the file it will run much faster. The file is also too large to upload to github.

## Requirements

- `preprocessed_tweets.csv`: A file of the tweets preproccessed. Other files can be used but this one is uploaded on the github. It just needs to be a csv file of the same format as the original tweets.csv.
- `logistic_model.py`: Python file required for the program to run.
- `one_hot_encoder.py`: Python file required to vectorize the tweets.

## Outputs

- `results.csv`: A csv file of three columns. The words in the dictionary; the sentiment the model assigned to them; and the confidence it has in that assignment.
- `vectors_file.txt`: Text file containing the converted tweets. Only made once.

# Preliminary Analysis Tool

## Usage

```bash
python preliminary_analysis_tool.py
```

It will then ask you for the file name of the tweets to analyse.

It will also ask you for the column name, how many bars to include, the way to sort them, and the minimum required tweets.

## Requirements

- `tweets.csv`: A file of tweets for the program to analyse.

## Outputs

- This program outputs two matplotlib plots. One chart of the counts of the given constraints, and one of the same graph as proportions.
