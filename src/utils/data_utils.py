import numpy as np


def remove_silence(audio, sr, threshold=-40, min_duration=0.1):
    """
    Remove silence from an audio signal based on amplitude threshold.

    Parameters:
    audio (numpy.ndarray): Input audio signal, as a 1-D NumPy array.
    sr (int): Sample rate of the input audio signal.
    threshold (float, optional): Amplitude threshold in dB for silence detection (default is -40).
    min_duration (float, optional): Minimum duration of silence in seconds (default is 0.2).

    Returns:
    numpy.ndarray: Audio signal with silence removed.
    """

    def detect_silence(audio, threshold):
        return np.where(np.abs(audio) >= 10 ** (threshold / 20))[0]

    non_silent_samples = detect_silence(audio, threshold)
    min_silent_samples = int(min_duration * sr)

    # Find continuous regions of silence
    silent_regions = np.split(non_silent_samples, np.where(np.diff(non_silent_samples) > min_silent_samples)[0] + 1)

    output_regions = [audio[region[0]:region[-1] + 1] for region in silent_regions]
    regions_idx = [(region[0], region[-1] + 1) for region in silent_regions]

    # Remove silence from the input audio
    output_audio = np.concatenate(output_regions)


    return output_regions, output_audio, regions_idx