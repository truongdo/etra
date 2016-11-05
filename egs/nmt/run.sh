#! /bin/bash
#
# run.sh
# Copyright (C) 2016 truong-d <truongdq54@gmail.com>
#
# Distributed under terms of the MIT license.
#


stage=0
gpu="auto"
pair=en2vi
data=data/$pair

. ./path.sh

if [[ ! -f data/nmt_example_en_vi.tar.gz ]]; then
  mkdir -p data
  (cd data && wget https://dl.dropboxusercontent.com/u/30873072/nmt_example_en_vi.tar.gz && \
    tar xvpfz nmt_example_en_vi.tar.gz)
fi

if [[ $stage -le 0 ]]; then
  steps/get_vocab.sh $data/train exp/lang_$pair || exit 1
fi

if [[ $stage -le 1 ]]; then
  for x in train dev;
  do
    steps/convert_data.sh --batch-size 128 exp/lang_$pair $data/$x exp/feat_$pair/$x || exit 1
  done
  steps/convert_data.sh --batch-size 128 exp/lang_$pair $data/test exp/feat_$pair/test_batch || exit 1   # this is for speed up evaluation time
  steps/convert_data.sh --batch-size 1 exp/lang_$pair $data/test exp/feat_$pair/test || exit 1
fi

if [[ $stage -le 2 ]]; then
  epoch=20
  emb_size=256
  lr_init=0.0001
  lr_stop=0.00001
  nj=1
  lr_scale=2
  hid_size=512
  drop_ratio=0.0
  depth=3
  prob_len=15
  optimizer="RMSpropGraves"
  dir=exp/model_${pair}_RMSpropGraves_h512_d3

  steps/train_nmt.sh --gpu $gpu --stage 0 --epoch $epoch \
    --nj $nj \
    --emb-size $emb_size --hid-size $hid_size \
    --lr-init $lr_init --lr-scale $lr_scale --lr-stop $lr_stop \
    --depth $depth --drop-ratio $drop_ratio \
    --optimizer $optimizer --prob-len $prob_len\
    --dev-dir exp/feat_$pair/dev exp/lang_$pair exp/feat_$pair/train $dir || exit 1
  
  # Note that to speed up the evaluation, we use minibatch for the test set
  # which requires padding end-of-sentence charater to sentences
  # in the same batch to make them the same length, this will lead to
  # higher error rate than normal. To have the actual error, run the
  # follow command with exp/feat_$pair/test, but it will take more time.
  echo "Evaluation --> $dir/test"
  $cmd $dir/log/test.log forward-nmt --gpu -1 \
        exp/feat_$pair/test_batch $dir $dir/test || exit 1

  steps/decode_nmt.sh --gpu -1 exp/lang_$pair exp/feat_$pair/test $dir $dir/decode
fi
