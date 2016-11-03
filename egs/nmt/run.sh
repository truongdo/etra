#! /bin/bash
#
# run.sh
# Copyright (C) 2016 truong-d <truongdq54@gmail.com>
#
# Distributed under terms of the MIT license.
#


stage=2
gpu=2
data=data/en-vi

. ./path.sh
if [[ $stage -le 0 ]]; then
  # TODO Test to make sure all words in vocab are unique
  steps/get_vocab.sh $data/train exp/lang || exit 1
fi

if [[ $stage -le 1 ]]; then
  for x in train dev;
  do
    steps/convert_data.sh --batch-size 128 exp/lang $data/$x exp/feat/$x || exit 1
  done
  steps/convert_data.sh --batch-size 1 exp/lang $data/test exp/feat/test || exit 1
  steps/convert_data.sh --batch-size 128 exp/lang $data/test exp/feat/test_batch || exit 1   # this is for speed up evaluation time
fi

if [[ $stage -le 2 ]]; then
  epoch=25
  emb_size=256
  lr_init=0.001
  lr_stop=0.0001
  lr_scale=2
  hid_size=1024
  drop_ratio=0.0
  depth=3
  dir=exp/model

  steps/train_nmt.sh --gpu $gpu --stage 7 --epoch $epoch \
    --emb-size $emb_size --hid-size $hid_size \
    --lr-init $lr_init --lr-scale $lr_scale --lr-stop $lr_stop \
    --depth $depth --drop-ratio $drop_ratio \
    --dev-dir exp/feat/dev exp/lang exp/feat/train $dir

  $cmd $dir/log/test.log forward-nmt --gpu -1 \
        exp/feat/test_batch $dir $dir/test || exit 1

  steps/decode_nmt.sh --gpu -1 exp/lang exp/feat/test $dir $dir/decode
fi
