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
nj=1

. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: steps/train_nmt.sh [options] <lang-dir> <train-data-dir> <out-dir>"
  echo "main options (for others, see top of script file)"
  echo "  --nj          (default 1)            # number of parallel jobs."
  echo "  --emb-size    (default 256)          # embedding size."
  echo "  --hid-size    (default 1024)         # hidden layer size."
  echo "  --depth       (default 3)            # depth of the encoder and decoder."
  echo "  --lr-init     (default 0.001)        # initial learning rate."
  echo "  --lr-scale    (default 2)            # learning rate descend scale."
  echo "  --lr-stop     (default 2)            # learning rate stopping threshold."
  echo "  --prob-len    (default 15)           # truncation length."
  echo "  --drop-ratio  (default 0.0)          # dropout ratio."
  echo "  --optimizer   (default SGD)          # optimizer SGD|RMSpropGraves|Adam|SMORMS3."
  echo "  --epoch       (default 20)           # number of training epoch."
  echo "  --dev-dir     (default None)         # development data set."
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

lang_dir=$1
dat_dir=$2
dir=$3

mkdir -p ${dir}/log
if [[ $stage -le 0 ]]; then
  src_vocab_size=`wc -l ${lang_dir}/src.vocab | cut -f 1 -d" "`
  trg_vocab_size=`wc -l ${lang_dir}/trg.vocab | cut -f 1 -d" "`
  echo "Initializing the model -> $dir"
  echo "-- src_vocab: $src_vocab_size  trg_vocab: $trg_vocab_size"
  echo "-- emb_size: $emb_size"

  mkdir -p ${dir}/mdl_e0/log
  $cmd ${dir}/mdl_e0/log/init.log init-nn-model --seed $seed \
      --drop-ratio $drop_ratio \
      --emb-size $emb_size --depth $depth --hid-size $hid_size \
      $src_vocab_size $trg_vocab_size $dir/mdl_e0 || exit 1
fi

prev_loss=100000000
dev_loss="None"
lr=$lr_init
rm $dir/training.log 2>/dev/null

mkdir -p ${dat_dir}/split$nj
paste $dat_dir/feat_src.scp $dat_dir/feat_trg.scp -d "|" | shuf > /tmp/feat_all.scp
cut -f 1 -d"|" /tmp/feat_all.scp > /tmp/feat_src.scp || exit 1
cut -f 2 -d"|" /tmp/feat_all.scp > /tmp/feat_trg.scp || exit 1
utils/split_data.sh --nj $nj --prefix feat_src /tmp/feat_src.scp ${dat_dir}/split$nj || exit 1
utils/split_data.sh --nj $nj --prefix feat_trg /tmp/feat_trg.scp ${dat_dir}/split$nj || exit 1

for (( e = 1; e < epoch + 1; e++ )); do
    in_mdl=$dir/mdl_e$(($e-1))
    o_mdl=$dir/mdl_e$e
    mkdir -p $o_mdl/log $o_mdl/dev

    start_time=`date +%s`
    if [[ $e -ge $stage ]]; then
      mdl_list=""
      pids=""
      for (( job = 1; job <= $nj; job++ )); do
        mkdir -p $o_mdl/$job
        (
        $cmd ${o_mdl}/$job/log/train-nmt.log train-nmt  \
                  --gpu $gpu --prob-len $prob_len \
                  --lr $lr --seed $e \
                  --optimizer $optimizer \
                  $dat_dir/split$nj/feat_src.${job}.scp $dat_dir/split$nj/feat_trg.${job}.scp \
                  $in_mdl $o_mdl/${job} || exit 1
        ) &
        pids+=" $!"
        sleep 2
        mdl_list="$mdl_list $o_mdl/${job}"
      done
      
      # Wait for background process
      for p in $pids; do
        if ! wait $p; then
          exit 1
        fi
      done
      
      $cmd ${o_mdl}/log/average.log average-nmt $mdl_list $o_mdl || exit 1
      if [[ ! -z $dev_dir ]]; then
          $cmd ${o_mdl}/log/dev.log forward-nmt --gpu $gpu \
                  $dev_dir ${o_mdl} ${o_mdl}/dev || exit 1
      fi
    fi
    end_time=`date +%s`
    train_loss=`tail -n 3 $o_mdl/1/log/train-nmt.log | head -n 1`
    dev_loss=`tail -n 3 $o_mdl/log/dev.log | head -n 1`

    echo "Epoch ${e} [$lr]: train_loss: $train_loss dev_loss: $dev_loss [`expr $end_time - $start_time`s]" | tee -a $dir/training.log

    if [[ ! -z $dev_dir ]]; then
      if [ $(echo "print $dev_loss > $prev_loss and $e > $warm_up" | python) == "True"  ]; then
          lr=`echo "print float($lr)/$lr_scale" | python`
          echo "Decending learning rate: $lr" | tee -a $dir/training.log
      fi
      prev_loss=$dev_loss
      if [ $(echo "print $lr < $lr_stop" | python) == "True"  ];
      then
          echo "Stop training" | tee -a $dir/training.log
          break
      fi
    fi
done
rm $dir/nnet.cfg $dir/nnet.mdl 2>/dev/null
cfg=`readlink -e $o_mdl/nnet.cfg`
mdl=`readlink -e $o_mdl/nnet.mdl`
if [[ ! -z $cfg ]]; then
    ln -s $cfg $dir/nnet.cfg
    ln -s $mdl $dir/nnet.mdl
else
    echo "Training is finished but cannot make the final model"
    exit 1
fi

