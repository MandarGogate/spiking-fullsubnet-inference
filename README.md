## Spiking-FullSubNet
[[Official GitHub Repo]](https://github.com/haoxiangsnr/spiking-fullsubnet)

### Minimal code for inference  

### Requirements
```bash
## Optional: create a new conda environment
conda create --name spiking-fullsubnet-inference python=3.10
conda activate spiking-fullsubnet-inference
## Install the required packages
pip install -r requirements.txt
```

## Usage
```bash
# model_zoo is one of the S, M, L, XL
# input_file is the path to the input sound file
# output_file is the path to the output sound file
python eval.py -i <input_file> -o <output_file> -m <model_zoo>
```
