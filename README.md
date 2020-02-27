Development
===========

## Requirements
python3.7
## Localhost

Run in python virtual environment
```
python -m pip install -r requirements.txt
python cosine_model.py train --logdir=/path/to/logs --training-noise=.1
```

## Localhost + Docker
```
docker build -t cosine .

docker run -v /path/to/logs:/logs cosine train --logdir /logs
```


## Tensorboard Visualizations
```
tensorboard --logdir=/path/to/logs
```
Note that you should have a environment with Tensorboard client installed.

## Todos
1. Hyperparameter tuning (lr, num_layers, activation function)
2. code for saving/loading model 
