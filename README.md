This project was used to test students at risk predictive models (the paper is still in review).

# Branches

- Use _OneSizeFitsAll-same-dataset_ branch to use other datasets as training data.
- Use _OneSizeFitsAll-trained-other-datasets_ branch to train each dataset separately by splitting each dataset in training and testing.

# Usage

- Copy your datasets to datasets/. The datasets names should follow ^dataset\d.csv pattern
- Tune moodle-keras-processes.py params
- Run moodle-keras-processes.py

```
python moodle-keras-processes.py --epochs=1 --run-prefix='a-test' --model-names="No context NN - 1 hidden.,With peers NN - 2 hidden." --test-datasets="dataset1,dataset2"
```

-Check the results in tensorboard (http://localhost:6006)


```
tensorboard --logdir=summaries 2> /dev/null & ; disown
chromium-browser http://localhost:6006
```
