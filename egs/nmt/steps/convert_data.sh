#! /bin/bash
#
# get_vocab.sh
# Copyright (C) 2016 Truong Do <truongdq54@gmail.com>
#
# Distributed under terms of the MIT license.
#
# This script convert word in text format into word id based
# on vocabularies extracted by using steps/get_vocab.sh

. ./path.sh

cmd=run.pl
batch_size=128

. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: steps/convert_data.sh [options] <lang-dir> <data-dir> <out-dir>"
  echo "main options (for others, see top of script file)"
  echo "  --batch-size    (default 128)  # minibatch size."
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

lang_dir=$1
dat_dir=$2
dir=$3

text=$dat_dir/text
src_vocab=$lang_dir/src.vocab
trg_vocab=$lang_dir/trg.vocab

[[ ! -f $text ]] && echo "Expecting $text file." && exit 1
[[ ! -f $src_vocab ]] && echo "Expecting $src_vocab file. Run steps/get_vocab.sh to obtain it" && exit 1
[[ ! -f $trg_vocab ]] && echo "Expecting $trg_vocab file. Run steps/get_vocab.sh to obtain it" && exit 1

mkdir -p $dir/log
echo "Converting $dat_dir into numeric values"
$cmd $dir/log/convert-data.log convert-data --batch-size $batch_size $src_vocab $trg_vocab $text $dir/feat_src.scp $dir/feat_trg.scp || exit 1
