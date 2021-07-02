
#!/usr/bin/env bash


mkdir -p data
pushd data

wget https://data.deepai.org/mnist.zip
unzip mnist

popd
