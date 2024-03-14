import argparse
from pathlib import Path
import soundfile as sf
import numpy as np
import toml
from model import Separator
import torch
import librosa


def enhance():
    global model
    config_path = Path(f"./model_zoo/{args.model}.toml").expanduser().absolute()
    model_path = Path(f"./model_zoo/{args.model}.pt").expanduser().absolute()
    config = toml.load(config_path.as_posix())
    model = Separator(**config["model_g"]["args"]).to(args.device)
    weights = torch.load(model_path.as_posix(), map_location=torch.device('cpu'))
    model.load_state_dict(weights)
    model.eval()
    noisy_audio = librosa.load(args.input, sr=16000)[0][np.newaxis, ...]
    noisy_audio = torch.from_numpy(noisy_audio)
    with torch.no_grad():
        enhanced_audio = model(noisy_audio).numpy().reshape(-1)
        sf.write(args.output, enhanced_audio, 16000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the model on a given audio file.')
    parser.add_argument('-i', '--input', help='Input audio file', required=True)
    parser.add_argument('-o', '--output', help='Output audio file', required=True)
    parser.add_argument('-m', '--model', help='Select model', required=True, default="XL",
                        choices=['S', 'M', 'L', 'XL'])
    parser.add_argument('-d', '--device', help='Device to run the model on', default='cpu')
    args = parser.parse_args()

    enhance()
