#! /bin/bash
#
# split_data.sh
# Copyright (C) 2016 truong-d <truong-d@truongd-ThinkPad-X1-Carbon-3rd>
#
# Distributed under terms of the MIT license.
#
# Split the data into chunks that will be processed later in parallel

# TODO do not split if the data is already exists

nj=4
prefix="data"

. ./path.sh
. parse_options.sh || exit 1
if [[ $# != 2 ]]; then
  echo "Usage: steps/split_data.sh [options] data.scp out_dir"
  echo "Option:"
  echo "--nj         number of jobs"
  echo "--prefix     Prefix of data"
fi

data=$1
dir=$2

mkdir -p $dir || exit 1
split_data=""
for n in $(seq $nj); do
  split_data="$split_data ${dir}/${prefix}.$n.scp"
done
split_scp.pl $data $split_data || exit 1;
