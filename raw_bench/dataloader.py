import argparse
import os

import pandas as pd
import torch
import torchaudio
import torch.utils.data as data
from loguru import logger

from .utils import AVERAGE_ENERGY_VCTK, read_test_df, split_test_chunks


class AudioDataset(data.Dataset):
    def __init__(
        self,
        dataset_filepath: str,
        datapath: dict,
        config: argparse.Namespace = None,
        num_samples: int = 16000,
        random_start_frame: bool = False,
        csv_delimiter: str = '|',
        mode: str = 'train',
        allow_missing_dataset: bool = False
    ):
        """
        Audio dataset loader for training, validation, and testing.

        Args:
            dataset_filepath: str
                Path to the dataset CSV file.
            datapath: dict
                Dictionary mapping dataset names to their root directories.
            config: argparse.Namespace, optional
                Configuration.
            num_samples: int, optional
                Number of samples per audio chunk.
            random_start_frame: bool, optional
                Whether to use a random start frame for each chunk.
            csv_delimiter: str, optional
                Delimiter used in the CSV file.
            mode: str, optional
                Mode of operation ('train', 'val', or 'test').
            allow_missing_dataset: bool, optional
                Whether to allow missing dataset.
        """
        self.dataset_filepath = dataset_filepath
        self.datapath = datapath
        self.config = config
        self.sr = config.sample_rate
        self.mono = config.mono
        self.num_samples = num_samples
        self.audio_chunk_duration = self.num_samples/self.sr
        if mode in ['train', 'val', 'test']:
            self.mode = mode
        else:
            raise ValueError(f"Entered mode {mode} not supported!")

        self.csv_delimiter = csv_delimiter
        self.allow_missing_dataset = allow_missing_dataset
        self.all_the_datasets_loaded = self._init_dataset_params()
        self.datasets_keys = sorted(list(set(self.datasets)))       
        self.cache = {}
        self.num_files = len(self.audio_filepaths)
        self.random_start_frame = random_start_frame and not self.mode == 'test'
       
    def __len__(
        self
    ) -> int:
        """
        Returns the number of audio files in the dataset.

        Returns:
            int: Number of files.
        """
        return self.num_files
        
    def load_audio_chunk(
        self,
        path: str,
        start_idx: int,
        orig_sr: int
    ) -> torch.Tensor:
        """
        Load and crop an audio chunk from a file.

        Args:
            path: str
                Path to audio file.
            start_idx: int
                Start index in audio file, in samples.
            orig_sr: int
                Original sample rate of the audio file.

        Returns:
            torch.Tensor: Audio tensor.
        """
        num_samples = int((orig_sr / self.sr) * self.num_samples) if orig_sr != self.sr else self.num_samples
        if self.mode == 'test':
            if (path, start_idx, num_samples) not in self.cache:
                audio_chunk, orig_sr = torchaudio.load(path, 
                                                       frame_offset=start_idx, 
                                                       num_frames=num_samples)
                
                if orig_sr != self.sr:
                    audio_chunk = torchaudio.transforms.Resample(orig_freq=orig_sr, 
                                                                 new_freq=self.sr)(audio_chunk)

                self.cache[(path, start_idx, num_samples)] = audio_chunk

            else:
                audio_chunk = self.cache[path]

        else:
            audio_chunk, orig_sr = torchaudio.load(path, 
                                                   frame_offset=start_idx, 
                                                   num_frames=num_samples)
                        
            if orig_sr != self.sr:                
                audio_chunk = torchaudio.transforms.Resample(orig_freq=orig_sr, 
                                                             new_freq=self.sr)(audio_chunk)
            
        if self.mono and audio_chunk.size(0) > 1:  # Check if audio has more than one channel
            audio_chunk = torch.mean(audio_chunk, dim=0, keepdim=True)  # Average channels to make mono

        # TODO: Do we really need this?
        audio_chunk *= torch.sqrt(AVERAGE_ENERGY_VCTK / torch.mean(audio_chunk**2))
        # Ensure `audio` is a 2D tensor by adding a batch dimension if necessary
        audio_chunk = audio_chunk.unsqueeze(0) if audio_chunk.dim() == 1 else audio_chunk
        
        return audio_chunk

    def _init_dataset_params(
        self
    )->bool:
        """
        Reads the dataset file and initializes file paths, start times, durations, and dataset names.
        
        Returns:
            bool: True if all the data is successfully loaded, False otherwise.
        """
        
        all_the_datasets_loaded = True
        
        if self.mode == 'train':
            df_dataset = pd.read_csv(self.dataset_filepath, delimiter=self.csv_delimiter)
        
        else:
            df_orig = read_test_df(test_df_path=self.dataset_filepath, delimiter=self.csv_delimiter)
            df_dataset = split_test_chunks(df_test=df_orig, chunk_duration=self.audio_chunk_duration)

        # Find problematic datapath entries
        missing_datapath = {key: value for key, value in self.datapath.items() if value is None or not os.path.exists(value)}
        missing_datapath = dict(sorted(missing_datapath.items()))
        
        if self.mode == 'train':
            if len(missing_datapath) > 0:
                raise ValueError(f"Missing datapath: {missing_datapath}")
        else:
            required_for_training = ['Tency', 'VCTK', 'BBC_Sound_Effects']
            missing_datapath = {key: value for key, value in missing_datapath.items() if key not in required_for_training}

            if len(missing_datapath) > 0:
                if not self.allow_missing_dataset:
                    error_msg = f"Missing datapath: {list(missing_datapath.keys())}." + \
                        "\n1. Please check `docs/datasets.md` for guidelines on how to set up the datasets." + \
                        "\n2. If you want to proceed even with missing datasets, please run with `allow_missing_dataset=true`"
                                     
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                all_the_datasets_loaded = False
                self.missing_datasets = list(missing_datapath.keys())
                logger.warning(f"Missing datasets: {self.missing_datasets}")
                logger.warning("Since there are some missing datasets, the test will be performed on the available datasets.")
                logger.warning("If you want to test on all datasets, please set the datapath for all datasets in the config file.")
                df_dataset = df_dataset[~df_dataset['dataset_name'].isin(self.missing_datasets)]                                        
                    
                if 'DEMAND' in self.missing_datasets:
                    logger.warning('DEMAND is required for background noise attack. This will be disabled.')
                    df_dataset = df_dataset[~df_dataset['attack_params'].astype(str).str.contains('noise_filepath', na=False)]

                if 'AIR' in self.missing_datasets:
                    logger.warning('AIR is required for reverb noise attack. This will be disabled.')
                    df_dataset = df_dataset[~df_dataset['attack_params'].astype(str).str.contains('rir_filepath', na=False)]

                assert len(df_dataset) > 0, (
                    "No available datasets found after applying allow_missing_dataset=True. "
                    "Please verify your dataset paths in `configs/datapath.yaml` and ensure at least one dataset is accessible."
                )
        dataset_name = df_dataset['dataset_name'].to_list()
        rel_filepaths = df_dataset['audio_filepath'].tolist()
        self.audio_filepaths = [f'{self.datapath[d]}/{r}' for d, r in zip(dataset_name, rel_filepaths)]

        assert all([os.path.exists(file) for file in self.audio_filepaths]), \
            "Some audio files do not exist in the specified paths. check if you set configs/datapath.yaml properly."
        for i, file in enumerate(self.audio_filepaths):
            assert os.path.exists(file), f"File does not exist: {i}:{file}"

        self.attack_types = df_dataset['attack_type'].tolist()
        self.attack_params = df_dataset['attack_params'].tolist()
        self.chunk_indices = df_dataset['chunk_idx'].tolist()
        self.start_times = df_dataset['start'].tolist()
        self.durations = df_dataset['duration'].tolist()
        self.datasets = df_dataset['dataset_name'].tolist() 

        return all_the_datasets_loaded

    def __getitem__(
        self,
        file_idx: int
    ):
        """
        Get a data sample from the dataset.

        Args:
            file_idx: int
                Index of the file to retrieve.

        Returns:
            tuple: Data sample (varies by mode).
        """
        if self.mode == 'train':
            carrier_filepath, start_time, duration, dataset = \
              self.audio_filepaths[file_idx], self.start_times[file_idx], self.durations[file_idx], self.datasets[file_idx]
            sample_rate = torchaudio.info(carrier_filepath).sample_rate

        else:
            carrier_filepath, start_time, duration, dataset, attack_type, attack_param, chunk_idx = \
              self.audio_filepaths[file_idx], self.start_times[file_idx], self.durations[file_idx], self.datasets[file_idx], \
              self.attack_types[file_idx], self.attack_params[file_idx], self.chunk_indices[file_idx]
            sample_rate = torchaudio.info(carrier_filepath).sample_rate
            start_idx = int(sample_rate * start_time)


        if self.random_start_frame:
            start_idx = int(torch.floor(torch.rand(1) * ((duration - self.audio_chunk_duration) * sample_rate)) + start_time * sample_rate) 
        else:
            start_idx = int(sample_rate * start_time)

        audio = self.load_audio_chunk(path=carrier_filepath,
                                      start_idx=start_idx,
                                      orig_sr=sample_rate)

        # Random attacks for train, for val and test, attack_type and parameters must be provided
        if self.mode == 'train':
            return audio, carrier_filepath, dataset

        else:
            return audio, carrier_filepath, dataset, attack_type, attack_param, chunk_idx, start_time
