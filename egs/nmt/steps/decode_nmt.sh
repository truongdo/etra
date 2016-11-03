#! /bin/bash
#
# train_nmt.sh
# Copyright (C) 2016 Truong Do <truongdq54@gmail.com>
#
# Distributed under terms of the MIT license.
#

. ./path.sh

cmd=run.pl
stage=0
prob_len=15
gpu=-1
epoch=20
seed=0
emb_size=256
lr_init=0.001
lr_stop=0.0001
optimizer="SGD"
lr_scale=2
hid_size=1024
warm_up=5
depth=3
drop_ratio=0.0
dev_dir=

. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: steps/decode_nmt.sh [options] <lang-dir> <test-data-dir> <model-dir> <out-dir>"
  echo "main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

lang_dir=$1
dat_dir=$2
model_dir=$3
dir=$4

mkdir -p $dir/log
if [[ $stage -le 0 ]]; then
  $cmd $dir/log/decode.log decode-nmt --gpu $gpu \
                $dat_dir $lang_dir ${model_dir} $dir || exit 1
fi
