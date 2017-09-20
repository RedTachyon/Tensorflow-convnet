# SharpestMinds Tensorflow Image Classification test

Everything is mostly according to the instructions, with the exception of some implementation details, like the classifier() function returning a keep_prob placeholder. But by running ./train.py you can train a convnet on the images dataset and validate them on the train/validation/test sets. 

Note: you need to have the data files (train_32x32.mat and test_32x32.mat) in the same directory as the repo, they're quite a bit too large to be uploaded to GitHub.

The neptune folder contains code that's ready to be used with Neptune, either locally or in cloud. Feel free to just run it as-is, but for magic, you can install Neptune and then just do 'neptune run' or 'neptune send --environment tensorflow-1.2-gpu-py3 --worker gcp-gpu-medium'
BTW. check Neptune out, it's awesome
