#!/bin/bash


timestamp=$(date +%s)

name=''  # given name for session [n]
disp_freq='100'  # frequency for updating the webpage [f]
intrm_num='5'  # number of intermediate frames [i]
port_num='8097'  # display port number [p]
gpu_ids=''  # ID of chosen GPU, this must not be set if CUDA_VISIBLE_DEVICES is set [g]
c_flag='false'  # run on CPU [c]
l_flag='false'  # print the running command without executing it [l]
weights_file=''  # a text file specifying different weights for different losses (see weight names in options/train_options.py) [w]
fine_size='192'  # working image resolution [s]
dataset='../datasets/boots'  # path to chosen dataset [d]
a_flag='false'  # if set, runs content-stye version [a]
t_flag='false'  # if set, runs with STN alignment [t]
batchsize='1'  # self-explanatory :) [b]

# ablation:
j_flag='false'  # no gan [j]
k_flag='false'  # no recon [k]
q_flag='false'  # no adain [q]
u_flag='false'  # no adj perc [u]
v_flag='false'  # no endp perc [v]



while getopts 'n:d:i:p:g:f:cs:w:latz:jkquv' flag; do
  case "${flag}" in
    n) name="${OPTARG}" ;;
    d) dataset="${OPTARG}" ;;
    f) disp_freq="${OPTARG}" ;;
    i) intrm_num="${OPTARG}" ;;
    p) port_num="${OPTARG}" ;;
    w) weights_file="${OPTARG}" ;;
    s) fine_size="${OPTARG}" ;;
    g) gpu_ids="${OPTARG}" ;;
    c) c_flag='true' ;;
    l) l_flag='true' ;;
    a) a_flag='true' ;;
    t) t_flag='true' ;;
    z) batchsize="${OPTARG}" ;;
    j) j_flag='true' ;;
    k) k_flag='true' ;;
    q) q_flag='true' ;;
    u) u_flag='true' ;;
    v) v_flag='true' ;;
  esac
done

weights=""

if [ "$weights_file" != '' ]
then
    tmp=`cat $weights_file`
    counter=1
    for tmp2 in $tmp
    do
	if [ `expr $counter % 2` -eq 1 ]
	then
	    weights="$weights --$tmp2"
	else
	    weights="$weights $tmp2"
	fi
	counter=$((counter+1))
    done
fi

load_size=$fine_size


dname=$(echo $dataset | sed s/.*\\///)
if [ "$name" != '' ]
then
    name2=$dname"_"$fine_size"_"$name
else
    name2=$dname"_"$fine_size
fi

name=$name2

args="train.py --dataroot $dataset --name $name --fineSize $fine_size --display_freq $disp_freq --display_id -1 --no_flip --nintrm $intrm_num $weights --batchSize $batchsize"

if [ "$port_num" == '-1' ]
then
    args="$args --display_id -1"
fi

if [ "$gpu_ids" != '' ]
then
    args="$args --gpu_ids $gpu_ids"
fi



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



if [ $j_flag == 'true' ]
then
    args="$args --no_gan"
fi

if [ $k_flag == 'true' ]
then
    args="$args --no_recon"
fi

if [ $q_flag == 'true' ]
then
    args="$args --no_adain"
fi

if [ $u_flag == 'true' ]
then
    args="$args --no_adjp"
fi

if [ $v_flag == 'true' ]
then
    args="$args --no_endpp"
fi

echo $args


if [ $l_flag == 'false' ]
then
    python3 $args
fi


