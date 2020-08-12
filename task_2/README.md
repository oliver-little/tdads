# Logistical Regressor

## Usage

```bash
python logistic_regressor.py
```
The first run of the program is the slowest. It has to generate a tweet vectors file which takes about five minutes on my laptop. 

Once it has the file it will run much faster. The file is also too large to upload to github.

## Requirements

- ```preprocessed_tweets.csv```: A file of the tweets preproccessed.

## Outputs
- ```results.csv```: A csv file of three columns. The words in the dictionary; the sentiment the model assigned to them; and the confidence it has in that assignment.
- ```vectors_file.txt```: Text file containing the converted tweets. Only made once. 

