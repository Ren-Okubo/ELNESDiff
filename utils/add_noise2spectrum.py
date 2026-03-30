import numpy as np


def add_noise2spectrum(spectrum, noise_strength_percent, seed):
    """
    スペクトルにガウシアンノイズを加える関数

    Parameters:
    spectrum (numpy.ndarray): ノイズを加える元のスペクトル
    noise_strength_percent (float): スペクトル最大値に対するノイズ強度（%）
    seed (int): 乱数シード

    Returns:
    numpy.ndarray: ノイズを加えたスペクトル
    """
    spectrum = np.asarray(spectrum)
    rng = np.random.default_rng(seed)

    noise_scale = np.max(spectrum) * noise_strength_percent / 100.0
    noise = rng.normal(loc=0.0, scale=noise_scale, size=spectrum.shape)

    return spectrum + noise
