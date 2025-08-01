# Test Dataset Download Instructions

To evaluate watermarking algorithms across diverse audio domains, we curated a comprehensive test dataset composed entirely of open-source audio collections. The dataset includes classical and popular music, speech, and environmental sounds, representing a wide range of real-world use cases.

## Key Features

- **Diverse Content**: Music (instrumental, vocal, ensemble), speech (studio and device-recorded), and environmental soundscapes.
- **Standardized Format**: All audio files are uniformly cropped into **6-second excerpts** to ensure consistent evaluation.
- **High Fidelity**: Audio is stored in raw, uncompressed format with a sample rate of **44.1 kHz**.
- **Open Licensing**: All components are sourced from publicly available datasets.

## Included Collections

Our test dataset is constructed from the following sources:

- **Bach10** ([GitHub](https://github.com/flippy-fyp/Bach10_v1.1))  
  Mixdowns of ten classical ensemble recordings of J.S. Bach pieces. We use only the stereo mix files.

- **Clotho** ([Zenodo](https://zenodo.org/records/3490684))  
  A diverse collection of short environmental sound clips. To ensure compatibility with file loaders, **we removed all whitespaces from filenames** during preprocessing.

- **DAPS (Device and Produced Speech)** ([Zenodo](https://zenodo.org/records/4660670))  
  A dataset of speech recorded in both studio and real-world device conditions. We include both high-quality reference tracks and corresponding device-recorded versions.

- **FreiSchuetz** ([AudioLabs](https://www.audiolabs-erlangen.de/resources/MIR/FreiDi/MultitrackDataset))  
  Multitrack and stereo recordings of three full-length opera performances, including professional mixes and raw microphone recordings. We use both the stereo mixes and selected multitrack stems.

- **GuitarSet** ([Zenodo](https://zenodo.org/records/3371780))  
  A collection of solo guitar recordings with detailed annotations of pitch, fretboard position, and playing technique. We use the mixdown audio tracks and ignore annotation files.

- **jaCappella** ([Hugging Face](https://huggingface.co/datasets/jaCappella/jaCappella))  
  A corpus of 50 Japanese a cappella vocal ensemble recordings. We use the stereo mixdowns and include all available isolated vocal tracks.

- **MAESTRO (v3.0.0)** ([Magenta](https://magenta.tensorflow.org/datasets/maestro#v300))  
  A dataset of paired audio and MIDI recordings from the International Piano-e-Competition. We use a **subset of the audio tracks** from MAESTRO-v3 for solo piano performance evaluation.

- **MoisesDB** ([Moises Research](https://music.ai/research/#datasets))  
  A multitrack dataset containing 240 musical tracks across 12 genres, performed by 45 artists. **We use only the stereo mixdowns**.  
  > 🔐 This dataset requires manual access approval. You must **submit a request to Moises** via their official website to obtain the data.

- **Piano Concerto Dataset (PCD)** ([AudioLabs](https://www.audiolabs-erlangen.de/resources/MIR/PCD/))  
  A dataset of stereo recordings from piano concertos. We use only a subset of the **raw solo piano tracks** (left-hand channel), without orchestra accompaniment.

## Access and Download

Due to licensing restrictions, we do **not** host or redistribute the raw audio files.  
Instead, we provide scripts to help you download and organize the datasets directly from their original sources.
Please note that it is users' responsibility to ensure compliance with all applicable licenses and terms of use for each dataset.


### `download_datasets.py`

Run the following command to automatically download and preprocess all datasets **except MoisesDB** (which requires manual access):

```shell
python download_datasets.py
```

#### Options

- `--tmp-dir`: Specify a directory for storing intermediate files during download and preprocessing (default: `tmp`).
- `--rm-tmp`: Remove the temporary directory after preprocessing is complete.

> **Note:** Downloading all datasets may take several hours or even days, depending on your internet speed and the dataset sizes.

> **Note** If you already have some of the required archives (e.g., ZIP, 7z, or TAR files), you can place them in the specified `tmp-dir` to speed up the process. The script will detect and use these files instead of downloading them again. For example:

```bash
user@desktop:~/raw_bench$ ls tmp
air_database_release_1_4.zip
audio_mono-mic.zip
maestro-v3.0.0.zip
clotho_evaluation.7z
PCD_2.0.1.zip
daps.tar.gz
```


### `prepare_moises.py`

To use MoisesDB, you must first [request and download the dataset](https://music.ai/research/#datasets) from the official website.  
Once you have obtained the files, run:

```shell
python prepare_moises.py --moisesdb-dir=PATH_TO_moisesdb_v0.1
```

This will preprocess the MoisesDB dataset for use with this repository.

## Connect datasets to the repository

Make sure that you set `configs/datapath/datapath.yaml` correctly.

```yaml
AIR: PATH_of_AIR
DEMAND: PATH_of_DEMAND
Clotho: PATH_of_Clotho
DAPS: PATH_of_DAPS
MoisesDB: PATH_of_MoisesDB
jaCappella: PATH_of_jaCappella
GuitarSet: PATH_of_GuitarSet
MAESTRO: PATH_of_MAESTRO
PCD: PATH_of_PCD
Bach10: PATH_of_Bach10
Freischuetz: PATH_of_Freischuetz
```

If you downloaded and processed your datasets using `download_datasets.py` and `prepare_moises.py`, set your `datapath.yaml` as shown below (this is the default configuration):

```yaml
AIR: test_data/AIR
DEMAND: test_data/DEMAND
Clotho: test_data/Clotho
DAPS: test_data/DAPS
MoisesDB: test_data/MoisesDB
jaCappella: test_data/jaCappella
GuitarSet: test_data/GuitarSet
MAESTRO: test_data/MAESTRO
PCD: test_data/PCD
Bach10: test_data/Bach10
Freischuetz: test_data/Freischuetz
```

## Metadata Files

Two CSV files provide metadata and configuration for each audio excerpt:

- `data/test_loose.csv`  
- `data/test_strict.csv`

Each row corresponds to one audio excerpt used in evaluation.  
The column **`orig_start`** indicates the starting time (in seconds) from which the n-second excerpt is cropped from the original file.

## Allow missing datasets

If you want to run using partial data, please run with `allow_missing_dataset=true`. For example, 

```bash
python scripts/eval.py \
    run_dir=runs/audioseal/pretrained_on_loose \
    ffmpeg4codecs=ffmpeg/ffmpeg-7.0.2-amd64-static/ffmpeg \
    allow_missing_dataset=true \
    --config-name=../configs/audioseal/eval_loose.yaml
```