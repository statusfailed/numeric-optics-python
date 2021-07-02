# Numeric Optics

A python library for constructing and training neural networks based on lenses
and reverse derivatives.

# Experiments / Usage Examples

Examples of model construction and training can be found in the
[./experiments](./experiments) directory.
To run them, you will need to install some additional dependencies:

    pip install -r experiments-requirements.txt

You can now run the following examples:

- [Iris dataset](http://archive.ics.uci.edu/ml/datasets/Iris/)
  - A single dense layer model: `python -m experiments.iris simple`
  - A neural network with a 20-node hidden layer: `python -m experiments.iris hidden`
- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)
  - A convolutional model `python -m experiments.convolutional`

## Data

To download the `iris` dataset, run

    ./get-iris-dataset.sh

To download the `mnist` dataset, run

    ./get-mnist-dataset.sh

## Equivalent Keras Experiments

For each experiment, we provide an equivalent written in [keras][keras].

[keras]: https://keras.io/

To run them, install keras and tensorflow dependencies:

    pip install -r keras-experiments-requirements.txt

And you can run each experiment as follows:

    # Single dense layer Iris model
    python -m keras_experiments.iris simple

    # 20-hidden-node neural network Iris model
    python -m keras_experiments.iris hidden

    # MNIST convolutional model
    python -m keras_experiments.convolutional

# Constructing a model

Models are built using composition `>>` and tensoring `@` of basic primitives.
For example, the lens representing a neural network dense layer is constructed as follows:

    assocL >> (identity @ linear) >> add >> activation

String-diagrammatically, that is:

    b -----------------\
                        \
    M ---\               [ + ] --- [ activation ] ----
          [ linear ] ---/
    x ---/

where:

- `M` is a matrix (the weights)
- `b` is a vector (the biases)
- `x` is a vector (the input to the network)
