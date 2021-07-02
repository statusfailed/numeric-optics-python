#!/usr/bin/env bash

mkdir -p data
pushd data

echo 'sepal_length,sepal_width,petal_length,petal_width,class' > iris.csv
wget http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
cat iris.data >> iris.csv

popd
