import io
import re
import subprocess
import tempfile
import torch
import torchaudio
from typing import Literal, Optional, Tuple


def get_mp3(wav_tensor: torch.Tensor, 
            sr: int, 
            bitrate: str = "128k",
            lowpass_freq: Optional[int] = None,
            ffmpeg4codecs: Optional[str] = None) -> torch.Tensor:
    """Convert a batch of audio files to MP3 format, maintaining the original shape.

    This function takes a batch of audio files represented as a PyTorch tensor, converts
    them to MP3 format using the specified bitrate, and returns the batch in the same
    shape as the input.

    Args:
        wav_tensor (torch.Tensor): Batch of audio files represented as a tensor.
            Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for MP3 conversion, default is '128k'.

    Returns:
        torch.Tensor: Batch of audio files converted to MP3 format, with the same
            shape as the input tensor.
    """
    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.view(1, -1).cpu()

    # Parse the bitrate value from the string
    match = re.search(r"\d+(\.\d+)?", bitrate)
    if match:
        parsed_bitrate = (
            match.group()
        )  # Default to 128 if parsing fails
    else:
        raise ValueError(f"Invalid bitrate specified (got {bitrate})")

    with tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as f_in, tempfile.NamedTemporaryFile(suffix=".mp3") as f_out:
        input_path, output_path = f_in.name, f_out.name

        # Save the tensor as a WAV file
        torchaudio.save(input_path, wav_tensor_flat, sr)

        # Prepare FFmpeg command for AAC conversion
        ffmpeg = "ffmpeg" if ffmpeg4codecs is None else ffmpeg4codecs 
        command = [
            ffmpeg,
            "-y",
            "-i",
            input_path,
            "-ar",
            str(sr),
            "-b:a",
            f"{parsed_bitrate}k",
            "-c:a",
            "libmp3lame",
        ]
        if lowpass_freq is not None:
            command += ["-cutoff", str(lowpass_freq)]
        command.append(output_path)

        # Run FFmpeg and suppress output
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Load the AAC audio back into a tensor
        mp3_tensor, _ = torchaudio.load(output_path)

    original_length_flat = batch_size * channels * original_length
    compressed_length_flat = mp3_tensor.shape[-1]

    # Trim excess frames
    if compressed_length_flat > original_length_flat:
        mp3_tensor = mp3_tensor[:, :original_length_flat]

    # Pad the shortened frames
    elif compressed_length_flat < original_length_flat:
        padding = torch.zeros(
            1, original_length_flat - compressed_length_flat, device=device
        )
        mp3_tensor = torch.cat((mp3_tensor, padding), dim=-1)

    # Reshape and adjust length to match original tensor
    wav_tensor = mp3_tensor.view(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]

    assert compressed_length == original_length, (
        "MP3-compressed audio does not have the same frames as original one. "
        "One reason can be ffmpeg is not  installed and used as proper backed "
        "for torchaudio, or the MP3 encoder is not correct. Run "
        "`torchaudio.utils.ffmpeg_utils.get_audio_encoders()` and make sure we see entry for"
        "MP3 in the output."
    )
    return wav_tensor.to(device)


def get_aac(
    wav_tensor: torch.Tensor,
    sr: int,
    bitrate: str = "128k",
    lowpass_freq: Optional[int] = None,
    ffmpeg4codecs: Optional[str] = None,
) -> torch.Tensor:
    """Converts a batch of audio tensors to AAC format and then back to tensors.

    This function first saves the input tensor batch as WAV files, then uses FFmpeg to convert
    these WAV files to AAC format. Finally, it loads the AAC files back into tensors.

    Args:
        wav_tensor (torch.Tensor): A batch of audio files represented as a tensor.
                                   Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for AAC conversion, default is '128k'.
        lowpass_freq (Optional[int]): Frequency for a low-pass filter. If None, no filter is applied.
        ffmpeg4codecs: (Optional[str]) = If none, use a defulat ffmpeg. Otherwise, use a specific ffmpeg.

    Returns:
        torch.Tensor: Batch of audio files converted to AAC and back, with the same
                      shape as the input tensor.
    """
    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Parse the bitrate value from the string
    match = re.search(r"\d+(\.\d+)?", bitrate)
    if match:
        parsed_bitrate = (
            match.group()
        )  # Default to 128 if parsing fails
    else:
        raise ValueError(f"Invalid bitrate specified (got {bitrate})")

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.view(1, -1).cpu() # one vary large audio file...

    with tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as f_in, tempfile.NamedTemporaryFile(suffix=".aac") as f_out:
        input_path, output_path = f_in.name, f_out.name

        # Save the tensor as a WAV file
        torchaudio.save(input_path, wav_tensor_flat, sr)

        # Prepare FFmpeg command for AAC conversion
        ffmpeg = "ffmpeg" if ffmpeg4codecs is None else ffmpeg4codecs 
        command = [
            ffmpeg,
            "-y",
            "-i",
            input_path,
            "-ar",
            str(sr),
            "-b:a",
            f"{parsed_bitrate}k",
            "-c:a",
            "aac",
        ]
        if lowpass_freq is not None:
            command += ["-cutoff", str(lowpass_freq)]
        command.append(output_path)

        # Run FFmpeg and suppress output
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Load the AAC audio back into a tensor
        aac_tensor, _ = torchaudio.load(output_path)

    original_length_flat = batch_size * channels * original_length
    compressed_length_flat = aac_tensor.shape[-1]

    # # Trim excess frames
    if compressed_length_flat > original_length_flat:
        min_distance = float('inf')
        min_index = -1
        length = wav_tensor.shape[-1]

        # Iterate over possible indices
        for index in range(aac_tensor.shape[-1] - length + 1):  # Sliding window
            # Extract the window from aac_tensor
            aac_window = aac_tensor[..., index:index+length].squeeze() 
            
            # Compute L1 distance
            l1_distance = torch.sum(torch.abs(aac_window - wav_tensor.cpu().squeeze()))
            
            # Update minimum distance and index
            if l1_distance < min_distance:
                min_distance = l1_distance
                min_index = index
            
        aac_tensor = aac_tensor[:, min_index:min_index+length]

    # Pad the shortened frames
    elif compressed_length_flat < original_length_flat:
        padding = torch.zeros(
            1, original_length_flat - compressed_length_flat, device=device
        )
        aac_tensor = torch.cat((aac_tensor, padding), dim=-1)

    # Reshape and adjust length to match original tensor
    wav_tensor = aac_tensor.view(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]

    assert compressed_length == original_length, (
        "AAC-compressed audio does not have the same frames as original one. "
        "One reason can be ffmpeg is not  installed and used as proper backed "
        "for torchaudio, or the AAC encoder is not correct. Run "
        "`torchaudio.utils.ffmpeg_utils.get_audio_encoders()` and make sure we see entry for"
        "AAC in the output."
    )
    return wav_tensor.to(device)


def get_vorbis(
    wav_tensor: torch.Tensor,
    sr: int,
    bitrate: str = '128k',
    lowpass_freq: Optional[int] = None,
    ffmpeg4codecs: Optional[str] = None,
 ) -> torch.Tensor:
     # see quality: https://en.wikipedia.org/wiki/Vorbis
    def convert_bitrate_to_quality(bitrate: str) -> int:
        if bitrate == '64k':
            return 0
        elif bitrate == '128k':
            return 4
        elif bitrate == '256k':
            return 8
        elif bitrate == '48k':
            return -1
        else:
            raise ValueError(f"Invalid bitrate: {bitrate}")

    import tempfile
    import subprocess

    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    quality = convert_bitrate_to_quality(bitrate)

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.view(1, -1).cpu() # one vary large audio file...

    with tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as f_in, tempfile.NamedTemporaryFile(suffix=".ogg") as f_out:
        input_path, output_path = f_in.name, f_out.name

        # Save the tensor as a WAV file
        torchaudio.save(input_path, wav_tensor_flat, sr)

        # Prepare FFmpeg command for AAC conversion
        ffmpeg = "ffmpeg" if ffmpeg4codecs is None else ffmpeg4codecs
        command = [
            ffmpeg,
            "-y",
            "-i",
            input_path,
            "-ar",
            str(sr),
            "-aq",
            str(quality),
            "-c:a",
            "libvorbis",
        ]
        if lowpass_freq is not None:
            command += ["-cutoff", str(lowpass_freq)]
        command.append(output_path)

        # Run FFmpeg and suppress output
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Load the AAC audio back into a tensor
        vorbis_tensor, _ = torchaudio.load(output_path)

    original_length_flat = batch_size * channels * original_length
    compressed_length_flat = vorbis_tensor.shape[-1]

    # Trim excess frames
    if compressed_length_flat > original_length_flat:
        vorbis_tensor = vorbis_tensor[:, :original_length_flat]

    # Pad the shortedn frames
    elif compressed_length_flat < original_length_flat:
        padding = torch.zeros(
            1, original_length_flat - compressed_length_flat, device=device
        )
        vorbis_tensor = torch.cat((vorbis_tensor, padding), dim=-1)

    # Reshape and adjust length to match original tensor
    wav_tensor = vorbis_tensor.view(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]

    assert compressed_length == original_length, (
        "OGG-compressed audio does not have the same frames as original one. "
        "One reason can be ffmpeg is not  installed and used as proper backed "
        "for torchaudio, or the OGG encoder is not correct. Run "
        "`torchaudio.utils.ffmpeg_utils.get_audio_encoders()` and make sure we see entry for"
        "OGG in the output."
    )
    return wav_tensor.to(device)



def mp3_wrapper(wav_tensor: torch.Tensor, 
                sr: int = 44100,
                bitrate = '64k', # 128k, 256k
                ffmpeg4codecs: str = None):
    
    return get_mp3(wav_tensor, sr=sr, bitrate=bitrate, ffmpeg4codecs=ffmpeg4codecs)


def aac_wrapper(wav_tensor: torch.Tensor, 
                sr: int,
                bitrate = '64k', # 128k, 256k
                ffmpeg4codecs: str = None):
    return get_aac(wav_tensor, sr, bitrate=bitrate, ffmpeg4codecs=ffmpeg4codecs)


def vorbis_wrapper(wav_tensor: torch.Tensor, 
                   sr: int,
                   bitrate: str = '64k',
                   ffmpeg4codecs: str = None):

    return get_vorbis(wav_tensor, sr, bitrate=bitrate, ffmpeg4codecs=ffmpeg4codecs)
