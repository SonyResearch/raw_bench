from collections import defaultdict
import hashlib
import numpy as np
from omegaconf import DictConfig, ListConfig
import pandas as pd
from pydub import AudioSegment
import torch
from torch.nn import functional as F


AVERAGE_ENERGY_VCTK = 0.002837200844477648


def batch_convolution(
    x: torch.Tensor, 
    f: torch.Tensor, 
    pad_both_sides: bool = True
    ) -> torch.Tensor:
    """
    Do batch-elementwise convolution between a batch of signals `x` and batch of filters `f`

    Args:
        x: torch.Tensor
            Input tensor of shape (batch_size x channels x signal_length).
        f: torch.Tensor
            Filter tensor of shape (batch_size x channels x filter_length).
        pad_both_sides: bool, optional
            Whether to zero-pad x on left and right or only on left (default: True).

    Returns:
        torch.Tensor: Convolved tensor
    """
    
    batch_size = x.shape[0]
    f = torch.flip(f, (2,))
    if pad_both_sides:
        x = F.pad(x, (f.shape[2]//2, f.shape[2]-f.shape[2]//2-1))
    else:
        x = F.pad(x, (f.shape[2]-1, 0))
    #TODO: This assumes single-channel audio, fine for now 
    return F.conv1d(x.view(1, batch_size, -1), f, groups=batch_size).view(batch_size, 1, -1)


def convert_pydub_to_torch(
    x: AudioSegment
) -> torch.Tensor:
    """
    Converts a Pydub AudioSegment object to a PyTorch tensor.

    Args:
        x: AudioSegment
            Input audio segment.

    Returns:
        torch.Tensor: Converted tensor.
    """
    samples = np.array(x.get_array_of_samples(), dtype=np.int16)
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
    x = torch.from_numpy(samples)[None, None]

    return x


def convert_torch_to_pydub(
    audio: torch.Tensor,
    sr: int = 44100
) -> AudioSegment:
    """
    Converts a PyTorch tensor to a Pydub AudioSegment object.

    Args:
        audio: torch.Tensor
            The input audio signal.
        sr: int, optional
            The sample rate of the audio signal (default: 44100).

    Returns:
        AudioSegment: The audio signal as a Pydub AudioSegment object.
    """
    waveform_np = audio.detach().cpu().numpy()

    # Pydub expects a single channel (mono) or interleaved multi-channel (stereo) data
    # Reshape and normalize the waveform
    if waveform_np.shape[0] > 1:
        # Interleave channels if it's multi-channel
        waveform_np = np.transpose(waveform_np).flatten()
    
    try:
    # Scale to int16 range (-32768 to 32767) for PCM format
        waveform_int16 = (waveform_np * 32767).astype(np.int16)
    except:
        waveform_np = np.sqrt(AVERAGE_ENERGY_VCTK / np.mean(waveform_np**2))
        waveform_int16 = (waveform_np * 32767).astype(np.int16)


    # Create AudioSegment from the raw data
    audio_segment = AudioSegment(
        data=waveform_int16.tobytes(),
        sample_width=2,  # 16-bit audio
        frame_rate=sr,
        channels=audio.shape[0]
    )

    return audio_segment


def split_test_chunks(
    df_test: pd.DataFrame,
    chunk_duration: float
) -> pd.DataFrame:
    """
    Splits test chunks into smaller chunks of duration `chunk_duration`
    and returns a DataFrame with the new chunk starts and durations.

    Args:
        df_test: pd.DataFrame
            DataFrame containing test data.
        chunk_duration: float
            Duration of each chunk in seconds to split the test audio files.

    Returns:
        pd.DataFrame: DataFrame containing test data with audio files split into chunks.
    """
    data_splitted = []

    for _, row in df_test.iterrows():
        duration = row['duration']
        start = row['start']
        
        #if duration // chunk_duration != duration / chunk_duration:
        #    raise ValueError(f'Audio duration is not a multiple of {chunk_duration}')

        for idx, chunk_start in enumerate(np.arange(start, start+duration, chunk_duration)):
            data_splitted.append([row['dataset_name'],
                                row['dataset_type'],
                                row['detail'],
                                row['file_id'],
                                row['audio_filepath'],
                                chunk_start,
                                chunk_duration,
                                row['sample_rate'],
                                row['attack_type'],
                                row['attack_params'],
                                idx])

    df_test_splitted = pd.DataFrame(data_splitted, 
                                columns=['dataset_name', 
                                         'dataset_type', 
                                         'detail', 
                                         'file_id',
                                         'audio_filepath',
                                         'start',
                                         'duration', 
                                         'sample_rate',
                                         'attack_type',
                                         'attack_params',
                                         'chunk_idx'])

    return df_test_splitted


def read_test_df(
    test_df_path: str,
    delimiter: str = '|'
) -> pd.DataFrame:
    """
    Reads test DataFrame from a file and returns a DataFrame where attack_params are read as a dictionary.

    Args:
        test_df_path: str
            Path to the test DataFrame file.
        delimiter: str, optional
            Delimiter used in the file (default: '|').

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    df_test = pd.read_csv(test_df_path, delimiter=delimiter)
    #df_test['attack_params'] = df_test['attack_params'].apply(json.loads)

    return df_test

def snr(
    orig: torch.Tensor,
    recon: torch.Tensor
):
    """
    Compute the signal-to-noise ratio (SNR) between two tensors.

    Args:
        orig: torch.Tensor
            Original tensor.
        recon: torch.Tensor
            Reconstructed tensor.

    Returns:
        torch.Tensor: SNR value.
    """
    N = orig.shape[-1] * orig.shape[-2]
    orig, recon = orig.cpu(), recon.cpu()
    rms1 = ((torch.sum(orig ** 2) / N) ** 0.5)
    rms2 = ((torch.sum((orig - recon) ** 2) / N) ** 0.5)
    snr = 10 * torch.log10((rms1 / rms2) ** 2)
    return snr


# copied from https://github.com/facebookresearch/audiocraft/blob/896ec7c47f5e5d1e5aa1e4b260c4405328bf009d/audiocraft/solvers/watermark.py#L697
def _bit_acc(
    decoded: torch.Tensor,
    original: torch.Tensor
):
    """
    Compute bit accuracy between decoded and original tensors.

    Args:
        decoded: torch.Tensor
            Decoded tensor.
        original: torch.Tensor
            Original tensor.

    Returns:
        torch.Tensor: Bit accuracy.
    """
    bit_acc = (decoded == original).float().mean()
    return bit_acc

# copied from https://github.com/facebookresearch/audiocraft/blob/896ec7c47f5e5d1e5aa1e4b260c4405328bf009d/audiocraft/solvers/watermark.py#L702
def compute_bit_acc(
    positive: torch.Tensor,
    original: torch.Tensor,
    mask: torch.Tensor = None
):
    """
    Compute bit accuracy.

    Args:
        positive: torch.Tensor
            Detector outputs [bsz, 2+nbits, time_steps].
        original: torch.Tensor
            Original message (0 or 1) [bsz, nbits].
        mask: torch.Tensor, optional
            Mask of the watermark [bsz, 1, time_steps].

    Returns:
        torch.Tensor: Bit accuracy.
    """
    decoded = positive[:, 2:, :]  # b 2+nbits t -> b nbits t
    if mask is not None:
        # cut last dim of positive to keep only where mask is 1
        new_shape = [*decoded.shape[:-1], -1]  # b nbits t -> b nbits -1
        decoded = torch.masked_select(decoded, mask == 1).reshape(new_shape)
    # average decision over time, then threshold
    decoded = decoded.mean(dim=-1) > 0  # b nbits
    return _bit_acc(decoded, original)


def get_saftdict(
    config
):
    """
    Recursively convert config objects to dictionaries or lists.

    Args:
        config: Any
            Configuration object.

    Returns:
        dict, list, or value: Converted configuration.
    """
    if isinstance(config, defaultdict) or config == {}:
        return None
    elif isinstance(config, (dict, DictConfig)):
        return {k: get_saftdict(v) for k, v in config.items()}
    elif isinstance(config, (list, ListConfig)):
        return [get_saftdict(i) for i in config]
    else:
        return config


def compute_mean_by_group(
    group_column,
    dataframe,
    metrics,
    key_columns: list = []
):
    """
    Compute the mean of specified metrics grouped by one or more columns.

    Args:
        group_column: str or list
            Column(s) to group by.
        dataframe: pd.DataFrame
            DataFrame containing the data.
        metrics: list
            List of metric column names to compute the mean for.
        key_columns: list, optional
            Additional columns to include in the output.

    Returns:
        pd.DataFrame: DataFrame with mean values grouped by the specified column(s).
    """
    key_columns += [metric for metric in metrics if metric not in key_columns]        
    if isinstance(group_column, str):
        mean_results = dataframe.groupby(group_column)[key_columns].mean()
        mean_results[group_column] = mean_results.index
        mean_results = mean_results.reset_index(drop=True)
        mean_results = mean_results[[group_column] + [col for col in mean_results.columns if col != group_column]]
    else:
        assert isinstance(group_column, (list, tuple, set)), "group_column must be a list, tuple or set"
        mean_results = dataframe.groupby(group_column)[key_columns].mean()
        for group in group_column:
            mean_results[group] = mean_results.index.get_level_values(group)
        mean_results = mean_results.reset_index(drop=True)
        mean_results = mean_results[group_column + [col for col in mean_results.columns if col not in group_column]]
    return mean_results

def init_worker(
    worker_id: int
):
    """
    Initialize worker process with a unique random seed.

    Args:
        worker_id: int
            Worker process ID.
    """
    np.random.seed(worker_id)

def file_checksum(path, algo='sha256', chunk_size=8192):
    h = hashlib.new(algo)
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()