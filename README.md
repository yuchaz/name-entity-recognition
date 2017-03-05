README.md file for CSE_517 assignment 4: Structured Learning with Feature-Rich Models.

###Run Best Model and print result
1. Run `source best_model.sh`
1. Run `source eval.sh`
1. Run `source read.sh`

###Data Location
1. First run `cp config.ini.template config.ini`
1. Change the `data_dir` attributes in `config.ini` file to the data location you place.
1. To simplified the procedure above, I have put the data dir inside this folder.

###Usage
1. Use `python main.py --help` to see argument list you can use.
1. Use `--lrn-rate [LEARN_RATE]` to specify learning rate eta of Perceptron.
1. Use `--epoch [EPOCH_TIME]` to specify how many epoch the algorithms will run through.
1. `--no-ctoken`, `--no-ptoken`, `--no-ftoken`,  `--no-cpos`, `--no-ppos`, `--no-fpos`, `--no-cchunk`, `--no-pchunk` and `--no-fchunk` means no current token, no previous token, no future token, no current pos tag, no previous pos tag, no future pos tag, no current syntactic chunk, no previous syntactic chunk and no future syntactic chunk as features, respectively.
1. Use `--test` to train the model with "train+dev" set and evaluate the model with "test" set. If not specified, it will be trained on "train" and evaluate the model with "dev" set.
1. When `--small` is specified, the learning algorithms use the truncated (small) dataset.

###Output
1. The output files will be located in the output folder with the filename:
    ```
    [dev|train]_eta[LEARN_RATE]_epoch_[EPOCH_TIME]_[FEATURE_CODE].txt
    ```
1. Use `./conlleval.txt < filename` to eval the result.

1. `FEATURE_CODE` is a sequence of digits of length six. It's the true-false value of `--no-ctoken`, `--no-ptoken`, `--no-ftoken`,  `--no-cpos`, `--no-ppos`, `--no-fpos`, `--no-cchunk`, `--no-pchunk` and `--no-fchunk`. Where 1 means True and 0 means False.
