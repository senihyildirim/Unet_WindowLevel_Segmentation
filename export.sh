#!/usr/bin/env bash

./build.sh

docker save bondbidhie2024_algorithm_senih | gzip -c > bondbidhie2024_algorithm_senih.tar.gz
