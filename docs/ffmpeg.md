# FFmpeg: Installation and Usage in raw-bench

[FFmpeg](https://ffmpeg.org/about.html) is the leading multimedia framework, capable of decoding, encoding, transcoding, muxing, demuxing, streaming, filtering, and playing virtually any audio or video format.

## Why do we need FFmpeg?

In `raw-bench`, FFmpeg is required for performing codec-based compression attacks. Specifically, we rely on the following codecs:

- **AAC**: `aac`
- **MP3**: `libmp3lame`
- **Vorbis**: `libvorbis`

> ⚠️ **Note:**  
> If you install FFmpeg via `conda`, it may not support all the necessary codecs listed above. Please verify that your FFmpeg build supports these codecs. Or see below.

## (Option 1) Recommended: Use a Pre-built FFmpeg for Codec Attacks

We recommend using a pre-built FFmpeg binary to ensure compatibility with all required codecs. You can first download a pre-build one, for example, [johnvansickle's](https://johnvansickle.com/ffmpeg/), and specify the path to this binary using the `ffmpeg4codecs` option when running evaluations:

```bash
python scripts/eval.py ffmpeg4codecs=/path/to/your/ffmpeg ...
```

This custom FFmpeg will be used **only** for codec-based attacks. All other audio processing (such as with `torchaudio`) will continue to use the system default.

If you do **not** specify the `ffmpeg4codecs` path, `raw-bench` will use the default FFmpeg available in your environment.

## (Option 2) Install FFmpeg by yourself

Of course you can install FFmpeg by yourself, and build the whole environment on top of it.

## Check if your ffmpeg supports all the codecs

If you run these,

```bash
your_ffmpeg -codecs | grep aac
your_ffmpeg -codecs | grep mp3
your_ffmpeg -codecs | grep vorbis
```
The proper compression algorithm must be listed as below. 
```bash
user@server:~/raw_bench$ your_ffmpeg -codecs | grep aac

ffmpeg version 6.1.1 Copyright (c) 2000-2023 the FFmpeg developers

...
DEAIL. aac                  AAC (Advanced Audio Coding) (decoders: aac aac_fixed)
D.AIL. aac_latm             AAC LATM (Advanced Audio Coding LATM syntax)
```