#!/bin/sh

version=$1

nvidia-docker build -t tcsfr:${version} .

nvidia-docker run -v /data/zhuoyu/tc_sfr/workspace/submission3_test/wdata/:/workdir -it tcsfr:${version} sh test.sh 2019-01-01,2019-02-02,2019-03-03,2019-08-21 /workdir/test.res

cd ../../../../scorer/

python scorer.py 2019-01-01,2019-02-02,2019-03-03,2019-08-21 ../data/2019wholeyear ../workspace/submission3_test/wdata/test.res
