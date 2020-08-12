# Recurrent Neural Network

## Usage

To test the RNN, with an already trained model "sentiment_analysis_model.h5", on the airline data supplied by Will:

```bash
python RNN.py [sentiment_analysis_model.h5]
```

To test the RNN, with an already trained model "sentiment_analysis_model.h5", on the extension airline data scraped from Twitter:

```bash
python RNN_other_datasets.py [sentiment_analysis_model.h5]
```

## Requirements

- ```tweets_for_RNN.csv```: A file containing all of the tweets (the extension tweets scraped from Twitter on top of those supplied by Will).

## Outputs

- ```sentiment_analysis_model_temp.h5```: If the already trained models are not supplied to the scripts, the program will train a new model (\~30 minutes) and save it in this file in the same directory.

## Troubleshooting

- ```Can't find model 'en'```: Please run your console as Administrator and run:

```bash
python -m spacy download en
```
