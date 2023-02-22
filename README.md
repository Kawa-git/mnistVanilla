# Mnist dataset in vanilla python
vanilla = python + numpy

## Installation
You may want to make a virtual environment to store numpy

```bash
virtualenv <virtualenv_name>
```

```bash
source <virtualenv_name>/bin/activate
```

```bash
pip install numpy
```
To download the datasets go to [this site](https://pjreddie.com/projects/mnist-in-csv/) or click [here](https://pjreddie.com/media/files/mnist_train.csv) for the train dataset and [here](https://pjreddie.com/media/files/mnist_test.csv) for the test dataset, put them in a folder called **dataset** inside the project file

## Usage
Just type:
```bash
python src/NN.py
```

**(Currently WIP)** 

Next updates: 
- saving weights
- adding biases
- running forward propagation on existing weights
