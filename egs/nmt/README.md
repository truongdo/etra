## How to use? ##
If you are in hurry to get a model or to test this out,
just run `./run.sh` and that's it.

## Tutorial ##
### Data preparation ###
To train an MT system, you need to prepare a parallel text corpus.
The data is structure as follows,

* data/train/text: the text file for training data.
* data/dev/text  : the text file for development data.
* data/test/text : the text file for testing data.

All text files have the format of "source|||target", for examples,
    ```
    how are you|||bạn có khỏe không
    ```

### run.sh ###
The script `run.sh` contains everything needed to train a MT system, after
understanding the idea of each command, just run `./run.sh` to build the system.

1. The first step is to get the vocabulary out from the training data text,
    ```bash
    if [[ $stage -le 0 ]]; then
      steps/get_vocab.sh $data/train exp/lang_$pair || exit 1
    fi
    ```

2. The second step is converting words in text files into word id format using
the extracted vocabulary. Note that we create 2 versions of test data (see `run.sh` for the reason).
    ```bash
    if [[ $stage -le 1 ]]; then
      for x in train dev;
      do
        steps/convert_data.sh --batch-size 128 exp/lang_$pair $data/$x exp/feat/$x || exit 1
      done
      steps/convert_data.sh --batch-size 128 exp/lang_$pair $data/test exp/feat/test_batch || exit 1   # this is for speed up evaluation time
      steps/convert_data.sh --batch-size 1 exp/lang_$pair $data/test exp/feat/test || exit 1
    fi
    ```

3. The final step is to train the system, you can find many parameters that you can config, such as training epoch (epoch), learning rate.
