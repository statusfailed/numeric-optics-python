# Numeric Optics

... A python library for constructing and training neural networks based on
lenses and reverse derivatives.

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

where `M` is a matrix (the weights), `b` is a vector (the biases), and `x` is
the input to the network.

# Demos

- `iris.py`: simple models for the [iris dataset][iris-dataset]
- `mnist.py`: (TODO)

[iris-dataset]: http://archive.ics.uci.edu/ml/datasets/Iris/
