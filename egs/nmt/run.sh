#! /bin/bash
#
# run.sh
# Copyright (C) 2016 truong-d <truongdq54@gmail.com>
#
# Distributed under terms of the MIT license.
#


stage=0
gpu=0
pair=vi2en
data=data/$pair

. ./path.sh
if [[ $stage -le 0 ]]; then
  steps/get_vocab.sh $data/train exp/lang_$pair || exit 1
fi

if [[ $stage -le 1 ]]; then
  for x in train dev;
  do
    steps/convert_data.sh --batch-size 128 exp/lang_$pair $data/$x exp/feat/$x || exit 1
  done
  steps/convert_data.sh --batch-size 128 exp/lang_$pair $data/test exp/feat/test_batch || exit 1   # this is for speed up evaluation time
  steps/convert_data.sh --batch-size 1 exp/lang_$pair $data/test exp/feat/test || exit 1
fi

if [[ $stage -le 2 ]]; then
  epoch=9
  emb_size=256
  lr_init=0.0001
  lr_stop=0.00001
  lr_scale=2
  hid_size=512
  drop_ratio=0.0
  depth=3
  prob_len=15
  optimizer="RMSpropGraves"
  dir=exp/model_${pair}_RMSpropGraves

  steps/train_nmt.sh --gpu $gpu --stage 0 --epoch $epoch \
    --emb-size $emb_size --hid-size $hid_size \
    --lr-init $lr_init --lr-scale $lr_scale --lr-stop $lr_stop \
    --depth $depth --drop-ratio $drop_ratio \
    --optimizer $optimizer --prob-len $prob_len\
    --dev-dir exp/feat/dev exp/lang_$pair exp/feat/train $dir
  
  # Note that to speed up the evaluation, we use minibatch for the test set
  # which requires padding end-of-sentence charater to sentences
  # in the same batch to make them the same length, this will lead to
  # higher error rate than normal. To have the actual error, run the
  # follow command with exp/feat/test, but it will take more time.
  $cmd $dir/log/test.log forward-nmt --gpu -1 \
        exp/feat/test_batch $dir $dir/test || exit 1

  steps/decode_nmt.sh --gpu -1 exp/lang_$pair exp/feat/test $dir $dir/decode
fi
