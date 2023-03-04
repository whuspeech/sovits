import os
import pyworld
import librosa
import numpy as np
import torch
from hubert import load_hubert


# extract fundamental frequency
def f0_sampling(f0_src, trg_len):
    f0_src = np.array(f0_src)
    src_len = len(f0_src)
    x_mid = np.arange(0, src_len * trg_len, src_len) / trg_len  # 在f0_src上采样trg_len个点
    x_left = x_mid - trg_len / src_len
    x_right = x_mid + trg_len / src_len
    differences = []

    f0_trg_mid = np.interp(x_mid, np.arange(src_len), f0_src)
    f0_trg_left = np.interp(x_left, np.arange(src_len), f0_src, left=0.0)
    f0_trg_right = np.interp(x_right, np.arange(src_len), f0_src, right=0.0)

    for i in range(trg_len):
        if i == 0:
            difference = f0_trg_right[i] - f0_trg_mid[i]
        elif i == trg_len - 1:
            difference = f0_trg_mid[i] - f0_trg_left[i]
        else:
            difference = f0_trg_right[i] - f0_trg_right[i]

        differences.append(difference)

    differences = np.array(differences, dtype=float)

    f0_trg_mid = f0_trg_mid.reshape(trg_len, 1)
    differences = differences.reshape(trg_len, 1)

    f0_trg = np.concatenate((f0_trg_mid, differences), axis=1)

    return f0_trg


def get_raw_f0(wav, sr):
    _f0, t = pyworld.dio(wav, sr)
    f0 = pyworld.stonemask(wav, _f0, t, sr)

    return f0


def process_dir(hubert_path, src_dir, dst_dir):
    # load hubert model
    hubert = load_hubert(hubert_path)
    wav_files = os.listdir(src_dir)
    for wav_file in wav_files:
        file_name = wav_file.replace('wav', 'npy')
        wav_path = os.path.join(src_dir, wav_file)
        save_path = os.path.join(dst_dir, file_name)
        wav, sr = librosa.load(wav_path)
        wav = librosa.resample(wav, sr, 22050)
        wav = librosa.to_mono(wav)

        wav_hubert = torch.from_numpy(wav).unsqueeze(0).unsqueeze(1)
        wav_f0 = wav.astype(np.double)

        unit = hubert.units(wav_hubert)
        f0 = get_raw_f0(wav_f0, 22050)
        f0 = f0_sampling(f0, unit.shape[1])

        feature = np.concatenate((unit, f0), axis=1)

        np.save(save_path, feature)


if __name__ == '__main__':
    pass
