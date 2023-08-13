Each folder represents a single experiment.

There is 2 item in each folder `train.py` and `iftest.py`:
- `train.py` will create 2 model alongside their evaluation, 1 before fine tune and 1 after fine tune
- `iftest.py` will try to output an inference time after the model have been created

`train.sh` will run every train.py
`iftest.sh` will run every iftest.py
