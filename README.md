# ASR project barebones

## Installation guide

< Write your installation guide here >

```shell
pip install -r ./requirements.txt
```

## Implementation guide

1) Search project for `raise NotImplementedError` and implement missing functionality
2) Make sure all tests work without errors
   ```shell
   python -m unittest discover hw_asr/tests
   ```

3) Make sure `test.py` works fine and works as expected.
   You should create files `default_test_config.json` and your installation guide should download your model
   checpoint and configs in `default_test_model/checkpoint.pth` and `default_test_model/config.json`.
   ```shell
   python test.py \
      -c default_test_config.json \
      -r default_test_model/checkpoint.pth \
      -t test_data \
      -o test_result.json
   ```
4) Use `train.py` for training

## Credits
this repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.