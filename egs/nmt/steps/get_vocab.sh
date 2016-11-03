#! /bin/bash
#
# get_vocab.sh
# Copyright (C) 2016 Truong Do <truongdq54@gmail.com>
#
# Distributed under terms of the MIT license.
#

. ./path.sh

cmd=run.pl
unk_thres=1
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
  echo "Usage: steps/get_vocab.sh [options] <data-dir> <out-dir>"
  echo "main options (for others, see top of script file)"
  echo "  --unk-thres     (default 1)    # Unknown threshold"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

dat_dir=$1
dir=$2

text=$dat_dir/text

[[ ! -f $text ]] && echo "Expecting $text file." && exit 1

mkdir -p $dir/log
echo "Obtaining vocab from $text"
$cmd $dir/log/get_vocab.log get_vocab --unk-thres $unk_thres $text $dir/src.vocab $dir/trg.vocab || exit 1
