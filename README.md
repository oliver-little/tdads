# TDADS

## Requirements
To install this project's dependencies, run `pip install -r requirements.txt`

## Usage

### Running Code

To run the code as-is, run `python main.py` from the main folder.

### Interpreting Results

Results of models are saved to `results.json`, which stores the confusion matrix, accuracy, training and prediction times.

## Configuration

Configuration is provided by `config.py`, in a dictionary format, generally:

```
models = {
	"model_name": {
		"fit": {
			[fit parameters as dict, if applicable]
		},
		"predict": {
			[predict parameters as dict, if applicable]
		},
	},
	...
}
```

To skip a model, simply add `"enabled": False` to its config dictionary.

Check the default config, config.py.default, for examples of possible parameters.
