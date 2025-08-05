# Test Dataset Download Instructions

To evaluate watermarking algorithms across diverse audio domains, we curated a comprehensive test dataset composed entirely of open-source audio collections. The dataset includes classical and popular music, speech, and environmental sounds, representing a wide range of real-world use cases.

## Key Features

- **Diverse Content**: Music (instrumental, vocal, ensemble), speech (studio and device-recorded), and environmental soundscapes.
- **Standardized Format**: All audio files are uniformly cropped into **6-second excerpts** to ensure consistent evaluation.
- **High Fidelity**: Audio is stored in raw, uncompressed format with a sample rate of **44.1 kHz**.
- **Open Licensing**: All components are sourced from publicly available datasets.

## Included Collections

Our test dataset is constructed from the following sources:


### [Bach10](https://github.com/flippy-fyp/Bach10_v1.1)

Mixdowns of ten classical ensemble recordings of J.S. Bach pieces. We use only the stereo mix files.

#### [License](https://github.com/flippy-fyp/Bach10_v1.1?tab=readme-ov-file#citing)

If you use the dataset in a work of your own that you wish to publish, please cite:

```bibtex
@article{DuanP11_SoundPrism_JSTSP,
  author    = {Zhiyao Duan and Bryan Pardo},
  title     = {Soundprism: An Online System for Score-Informed Source Separation of Music Audio},
  journal   = {{IEEE} Journal of Selected Topics in Signal Process.},
  year      = {2011},
  volume    = {5},
  number    = {6},
  pages     = {1205--1215}
}
```

### [Clotho](https://zenodo.org/records/3490684)

A diverse collection of short environmental sound clips. To ensure compatibility with file loaders, **we removed all whitespaces from filenames** during preprocessing.

#### [License](https://zenodo.org/records/3490684)
> under the corresponding licences (mostly CreativeCommons with attribution) of Freesound [1] platform, mentioned explicitly in the CSV files for each of the audio files. That is, each audio file in the 7z archives is listed in the CSV files with the meta-data. 

#### Reference

If you use Clotho, please cite our paper.

```bibtex
@inproceedings{DrossosLV20_Clotho_ICASSP,
  author    = {Konstantinos Drossos and Samuel Lipping and Tuomas Virtanen},
  title     = {{C}lotho: {A}n Audio Captioning Dataset}, 
  booktitle = {Proc. of the {IEEE} Int. Conf. on Acoust., Speech, and Signal Process. ({ICASSP})},
  year      = {2020},
  pages     = {736--740},
  doi       = {10.1109/ICASSP40776.2020.9052990}
}
```

### [DAPS](https://zenodo.org/records/4660670) (Device and Produced Speech)  

A dataset of speech recorded in both studio and real-world device conditions. We include both high-quality reference tracks and corresponding device-recorded versions.

#### [License]((https://zenodo.org/records/4660670))

Creative Commons Attribution Non Commercial 4.0 International

### Reference

```bibtex
@article{Mysore15_DAPS_SPS,
  author    = {Gautham J. Mysore},
  title     = {Can we Automatically Transform Speech Recorded on Common Consumer Devices in Real-World Environments into Professional Production Quality Speech? -- {A} Dataset, Insights, and Challenges}, 
  journal   = {{IEEE} Signal Process. Lett.}, 
  year      = {2015},
  volume    = {22},
  number    = {8},
  pages     = {1006--1010},
  doi       = {10.1109/LSP.2014.2379648},
}
```

### [FreiSchuetz](https://www.audiolabs-erlangen.de/resources/MIR/FreiDi/MultitrackDataset)

Multitrack and stereo recordings of three full-length opera performances, including professional mixes and raw microphone recordings. We use both the stereo mixes and selected multitrack stems.

#### [Lisence](https://www.audiolabs-erlangen.de/resources/MIR/FreiDi/MultitrackDataset)

This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

#### Reference

```bibtex
@inproceedings{PraetzlichMBV15_FreiDi_ISMIR-LBD,
  author    = {Thomas Pr{\"a}tzlich and Meinard M{\"u}ller and Benjamin W. Bohl and Joachim Veit},
  title     = {{F}reisch{\"u}tz {D}igital: Demos of audio-related contributions},
  booktitle = {Demos and Late Breaking News of the Int. Soc. for Music Inf. Retriev. Conf. ({ISMIR})},
  address   = {M{\'a}laga, Spain},
  year      = {2015},
}
```

### [GuitarSet](https://zenodo.org/records/3371780)

A collection of solo guitar recordings with detailed annotations of pitch, fretboard position, and playing technique. We use the mixdown audio tracks and ignore annotation files.

#### [License](https://zenodo.org/records/3371780)

Creative Commons Attribution 4.0 International

#### Reference

If you make use of GuitarSet for academic purposes, please cite the following publication:

```bibtex
@inproceedings{XiEtAl18_GuitarSet_ISMIR,
  title     = {GuitarSet: A Dataset for Guitar Transcription.},
  author    = {Qingyang Xi and Rachel Bittner and Johan Pauwels and Xuzhou Ye and Juan Pablo Bello},
  booktitle = {Proc. of the Int. Soc. for Music Inf. Retriev. Conf. ({ISMIR})},
  address   = {Paris, France},
  pages     = {453--460},
  year      = {2018}
}
```

### [jaCappella](https://huggingface.co/datasets/jaCappella/jaCappella)

A corpus of 50 Japanese a cappella vocal ensemble recordings. We use the stereo mixdowns and include all available isolated vocal tracks.

#### [License](https://tomohikonakamura.github.io/jaCappella_corpus/)

Please refer to [this](https://tomohikonakamura.github.io/jaCappella_corpus/)

#### Reference

```bibtex
@inproceedings{NakamuraEtAl23_jaCapella_ICASSP,
  author    = {Tomohiko Nakamura and Shinnosuke Takamichi and Naoko Tanji and Satoru Fukayama and Hiroshi Saruwatari},
  title     = {ja{C}appella corpus: A {J}apanese a cappella vocal ensemble corpus},
  booktitle = {Proc. of the {IEEE} Int. Conf. on Acoust., Speech, and Signal Process. ({ICASSP})},
  year      = {2023},
  doi       = {10.1109/ICASSP49357.2023.10095569},
}
```

### [MAESTRO](https://magenta.tensorflow.org/datasets/maestro#v300) (v3.0.0)

A dataset of paired audio and MIDI recordings from the International Piano-e-Competition. We use a **subset of the audio tracks** from MAESTRO-v3 for solo piano performance evaluation.

#### [License](https://magenta.tensorflow.org/datasets/maestro#license)

The dataset is made available by Google LLC under a Creative Commons Attribution Non-Commercial Share-Alike 4.0 (CC BY-NC-SA 4.0) license.

#### Reference

```bibtex
@inproceedings{
  hawthorne2018enabling,
  title={Enabling Factorized Piano Music Modeling and Generation with the {MAESTRO} Dataset},
  author={Curtis Hawthorne and Andriy Stasyuk and Adam Roberts and Ian Simon and Cheng-Zhi Anna Huang and Sander Dieleman and Erich Elsen and Jesse Engel and Douglas Eck},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=r1lYRjC9F7},
}
```

### [MoisesDB](https://music.ai/research/#datasets)  

A multitrack dataset containing 240 musical tracks across 12 genres, performed by 45 artists. **We use only the stereo mixdowns**.  

  > This dataset requires manual access approval. You must **submit a request to Moises** via their official website to obtain the data.

#### [License](https://github.com/moises-ai/moises-db/blob/main/LICENSE)

MoisesDB is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).

#### Reference

```bibtex
@misc{pereira2023moisesdb,
      title={Moisesdb: A dataset for source separation beyond 4-stems}, 
      author={Igor Pereira and Felipe Araújo and Filip Korzeniowski and Richard Vogl},
      year={2023},
      eprint={2307.15913},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```

### Piano Concerto Dataset ([PCD](https://www.audiolabs-erlangen.de/resources/MIR/PCD/))  

A dataset of stereo recordings from piano concertos. We use only a subset of the **raw solo piano tracks** (left-hand channel), without orchestra accompaniment.

#### License

#### Reference

```bibtex
@article{OezerSALSM23_PCD_TISMIR,
  title     = {{P}iano {C}oncerto {D}ataset {(PCD)}: A Multitrack Dataset of Piano Concertos},
  author    = {Yigitcan {\"O}zer and Simon Schw{\"a}r and Vlora Arifi-M{\"u}ller and Jeremy Lawrence and Emre Sen and Meinard M{\"u}ller},
  journal   = {Trans. of the Int. Soc. for Music Inf. Retriev. ({TISMIR})},
  volume    = {6},
  number    = {1},
  year      = {2023},
  pages     = {75--88},
  doi       = {10.5334/tismir.160},
}
```

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