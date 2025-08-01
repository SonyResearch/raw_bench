from pydub.effects import compress_dynamic_range
from pydub.utils import audioop, db_to_float, ratio_to_db

import torch

from ..utils import convert_torch_to_pydub, convert_pydub_to_torch


def dynamic_range_compression(audio: torch.Tensor,
                              threshold: float = -20,
                              ratio: float = 4.0,
                              sr: int = 44100,
                              **kwargs) -> torch.Tensor:
    """Wrapper for applying dynamic compression to an audio signal with pydub
    
    Parameters
    ----------
    audio: torch.Tensor
        The input audio signal.
    
    threshold: float
        The threshold level for compression (in dB).

    sr: int
        The sample rate of the audio signal.
   
    Returns
    -------
    compressed_audio: torch.Tensor
        The compressed audio signal.
    """
    init_device = audio.device


    # Peak-normalize the audio signal
    max_val = torch.max(torch.abs(audio))
    audio_norm = audio / max_val
    audio_seg = convert_torch_to_pydub(audio=audio_norm, sr=sr)
    
    compressed_audio_seg = compress_dynamic_range(audio_seg,
                                                  ratio=ratio, 
                                                  threshold=threshold,
                                                  **kwargs)
    
    compressed_audio = convert_pydub_to_torch(compressed_audio_seg).to(init_device)

    # De-normalize the compressed audio signal
    compressed_audio *= max_val

    return compressed_audio


def dynamic_range_expansion(audio: torch.Tensor,
                            threshold: float = -20,
                            ratio: float = 4.0,
                            sr: int = 44100,
                            **kwargs) -> torch.Tensor:
    init_device = audio.device

    # Peak-normalize the audio signal
    max_val = torch.max(torch.abs(audio))
    audio_norm = audio / max_val

    audio_seg = convert_torch_to_pydub(audio=audio_norm,
                                       sr=sr)
    
    expanded_audio_seg = expand_dynamic_range(audio_seg,
                                              ratio=ratio, 
                                              threshold=threshold,
                                              **kwargs)
    expanded_audio = convert_pydub_to_torch(expanded_audio_seg).to(init_device)

    # De-normalize the expanded audio signal
    expanded_audio *= max_val

    return expanded_audio


def expand_dynamic_range(seg, 
                         threshold=-20.0, 
                         ratio=4.0, 
                         attack=5.0, 
                         release=50.0):
    """
    Keyword Arguments:
        
        threshold - default: -20.0
            Threshold in dBFS. default of -20.0 means -20dB relative to the
            maximum possible volume. 0dBFS is the maximum possible value so
            all values for this argument sould be negative.

        ratio - default: 4.0
            Expansion ratio. Audio louder than the threshold will be 
            expanded to ratio the volume. A ratio of 4.0 is equivalent to
            a setting of 4:1 in a pro-audio compressor like the Waves C1.
        
        attack - default: 5.0
            Attack in milliseconds. How long it should take for the compressor
            to kick in once the audio has exceeded the threshold.

        release - default: 50.0
            Release in milliseconds. How long it should take for the compressor
            to stop compressing after the audio has falled below the threshold.

    
    For an overview of Dynamic Range Compression, and more detailed explanation
    of the related terminology, see: 

        http://en.wikipedia.org/wiki/Dynamic_range_compression
    """

    thresh_rms = seg.max_possible_amplitude * db_to_float(threshold)
    
    look_frames = int(seg.frame_count(ms=attack))
    def rms_at(frame_i):
        return seg.get_sample_slice(frame_i - look_frames, frame_i).rms
    def db_over_threshold(rms):
        if rms == 0: return 0.0
        db = ratio_to_db(rms / thresh_rms)
        return max(db, 0)

    output = []

    # amount to reduce the volume of the audio by (in dB)
    amplification = 0.0
    
    attack_frames = seg.frame_count(ms=attack)
    release_frames = seg.frame_count(ms=release)
    for i in range(int(seg.frame_count())):
        rms_now = rms_at(i)
        
        # with a ratio of 4.0 this means the volume will exceed the threshold by
        # 1/4 the amount (of dB) that it would otherwise
        max_amplification  = (ratio - 1) * db_over_threshold(rms_now)
        
        amplification_inc = max_amplification / attack_frames
        amplification_dec = max_amplification / release_frames
        
        if rms_now > thresh_rms and amplification <= max_amplification:
            amplification += amplification_inc
            amplification = min(amplification, max_amplification)
        else:
            amplification -= amplification_dec
            amplification = max(amplification, 0)
        
        frame = seg.get_frame(i)
        if amplification != 0.0:
            frame = audioop.mul(frame,
                                seg.sample_width,
                                db_to_float(-amplification))
        
        output.append(frame)
    
    return seg._spawn(data=b''.join(output))
