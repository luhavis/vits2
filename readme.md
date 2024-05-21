# VITS2
This repo is practice project.

# Tensorboard
```bash
tensorboard --logdir {model_dir} --port 6006
```

# Prepare train, validation data
```bash
# single speaker
python preprocess.py --text_index 1 --filelists filelists/train.txt filelists/train_val.txt --text_cleaners korean_cleaners

# multi speaker
python preprocess.py --text_index 2 --filelists filelists/train.txt filelists/train_val.txt --text_cleaners korean_cleaners
```

# Train
```bash
python train_ms.py --config {config_file_path} --model {model_name}
```

# Setup env

python 3.10 require


### Create anaconda env
```bash
conda create -n vits2 python=3.10
```

### Activate anaconda env
```bash
conda activate vits2
```

### Install packages
```bash
pip install poetry
poetry install
```

### Build monotonic_align
```bash
cd monotomic_align
mkdir monotomic_align
python setup.py build_ext --inplace
```

# References
- [daniilrobnikov/vits2](https://github.com/daniilrobnikov/vits2)
- [p0p4k/vits2_pytorch](https://github.com/p0p4k/vits2_pytorch)
- [lucidrains/BS-RoFormer](https://github.com/lucidrains/BS-RoFormer)
- [ZFTurbo/Music-Source-Separation-Training](https://github.com/ZFTurbo/Music-Source-Separation-Training)