#!/bin/bash


timestamp=$(date +%s)

name=''  # given name for test session [n]
intrm_num='5'  # number of intermediate frames [i]
c_flag='false'  # run on CPU [c]
l_flag='false'   # print the running command without executing it [l]
fine_size='192'  # working image resolution [s]
dataset='../datasets/boots'  # path to chosen dataset [d]
a_flag='false'  # if set, runs content-stye version [a]
t_flag='false'  # if set, runs with STN alignment [t]
batchsize='1'  # [b]
dir=''  # folder containing trained model (assumes it is under ./checkpoints) [v]

q_flag='false'  # no adain (ablation) [q]


while getopts 'n:d:i:gcs:latz:v:q' flag; do
  case "${flag}" in
    n) name="${OPTARG}" ;;
    d) dataset="${OPTARG}" ;;
    i) intrm_num="${OPTARG}" ;;
    s) fine_size="${OPTARG}" ;;
    c) c_flag='true' ;;
    l) l_flag='true' ;;
    a) a_flag='true' ;;
    t) t_flag='true' ;;
    z) batchsize="${OPTARG}" ;;
    v) dir="${OPTARG}" ;;
    q) q_flag='true' ;;
  esac
done



args="test.py --dataroot $dataset --name $name --save_dir $dir --fineSize $fine_size --nintrm $intrm_num --batchSize $batchsize"


if [ $c_flag == 'true' ]
then
    args="$args --gpu_ids -1"
fi

if [ $a_flag == 'true' ]
then
    args="$args --costl"
fi

if [ $t_flag == 'true' ]
then
    args="$args --stn"
fi

if [ $q_flag == 'true' ]
then
    args="$args --no_adain"
fi

echo $args


if [ $l_flag == 'false' ]
then
    python3 $args
fi
