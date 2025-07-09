from datetime import datetime
import io
from mel_cepstral_distance import compare_audio_files
from loguru import logger
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
import random
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import DataLoader
from typing import Dict, Union
import wandb

from ..attacks import AudioAttack
from ..custom_stft import STFT
from ..dataloader import AudioDataset
from ..logger import ExperimentLogger
from ..utils import compute_mean_by_group, init_worker


class Solver(object):
    """
    Base class for audio watermarking solvers.

    Handles data loading, logging, and many utility functions for the evaluation.
    """    
    def __init__(self, 
                 config: Union[Path, DictConfig]):
        """
        Initialize the Solver with configuration and set up data loaders, logging, and experiment directories.

        Args:
            config (DictConfig): Configuration object for the experiment.
        """        
        # Load config
        if isinstance(config, (str, Path)):
            self.config = OmegaConf.load(config)
        elif isinstance(config, DictConfig):
            self.config = config
        else:
            raise TypeError("Config must be a DictConfig, str, or Path.")

        # Device
        if torch.cuda.is_available() and str(config.device).startswith('cuda'):
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # Log if fallback happened
        if str(self.device) != str(config.device):
            logger.info(f"Using device: {self.device} (requested: {config.device})")
        
        # Set model_type
        try:
            self.model_type = config.model_type
            if self.model_type is None:
                raise ValueError("config.model_type is set to None. This value must be provided.")
        except AttributeError:
            raise AttributeError("config must include a 'model_type' field. This value is required.")
    
        # Create a run_dir with model name and timestamp, if not available
        self.run_dir = config.get("run_dir")
        if self.run_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.run_dir = f"{self.model_type}/{timestamp}"

        self.test_results_dir = os.path.join(self.run_dir, 'results')
        os.makedirs(self.test_results_dir, exist_ok=True)
        
        attack_level = config.get("test_suffix", "unknown")
        random_seed = config.get("random_seed_for_eval", 42)
        
        if not hasattr(config, "exp_name") or config.exp_name is None:
            self.config.exp_name = f"{self.model_type}_{attack_level}_seed{random_seed}"

        exp_name = self.config.exp_name

        self.exp_logger = ExperimentLogger(self.run_dir, 
                           use_wandb=config.wandb,
                           project_name=config.get("project_name", "raw-benchmark"),
                           exp_name=exp_name)
        
        self.csv_delimiter = config.get("csv_delimiter", "|")
        self.test_path = config.dataset.test_path

        if config.full_perceptual:
            self.init_visqol()

        # Core audio parameters
        try:
            self.sample_rate = config.sample_rate
            self.eval_seg_duration = config.dataset.eval_seg_duration
        except AttributeError as e:
            raise AttributeError(f"Missing required config field: {e}")

        self.num_samples = int(self.eval_seg_duration * self.sample_rate)

        # Optional STFT setup
        stft_cfg = config.get("stft", None)
        if stft_cfg is not None:
            self.stft = STFT(
                filter_length=stft_cfg.n_fft,
                hop_length=stft_cfg.hop_len,
                win_length=stft_cfg.win_len,
                num_samples=self.num_samples
            ).to(self.device)
        else:
            self.stft = None

        # For training, for val and test these are always active
        self.audio_attack = AudioAttack(sr=self.sample_rate,
                                        datapath=self.config.datapath, #TODO
                                        stft=self.stft,
                                        mode='test',
                                        config=config.attack,
                                        ffmpeg4codecs=config.ffmpeg4codecs,
                                        device=self.device)
        
        # Set number of data loading workers (default: 4)
        self.num_workers = config.get("num_workers", 4)

        self.build_dataloaders()
        
        # logging
        os.makedirs(self.run_dir + '/logs', exist_ok=True)
        logger.add(os.path.join(self.run_dir + '/logs', "stdout.log"))

    def build_dataloaders(self):
        """
        Build PyTorch DataLoaders for test dataset based on configuration.
        """
        if self.test_path is not None:
            assert os.path.isfile(self.test_path), f"Test path {self.test_path} does not a valid file."
            # Extract settings with default fallbacks and error handling
            test = AudioDataset(
                self.test_path,
                self.config.datapath,
                self.config.dataset,
                num_samples=self.num_samples,
                mode='test',
                allow_missing_dataset=self.config.allow_missing_dataset
            )

            if self.config.dataset.eval_batch_size > 1:
                logger.warning("Currently, we only support batch size of 1 for test DataLoader. ")

            self.test_loader = DataLoader(
                test,
                batch_size=1,  # self.config.dataset.eval_batch_size
                shuffle=False,
                num_workers=self.num_workers,
                worker_init_fn=init_worker
            )

            logger.info(f"Test loader built with {len(test)} samples.")

        else:
            logger.error("No test path specified. Test DataLoader cannot be created.")
            raise ValueError("Missing required test_path.")

    def close_eval(self):
        """
        Close the solver, releasing any resources or handles.
        """
        logger.info("Evaluation was done successfully. Closing solver and releasing resources.")
        if hasattr(self, 'exp_logger'):
            self.exp_logger.close()
        if not self.test_loader.dataset.all_the_datasets_loaded:
            logger.warning("This test is only done for the partial datasets (allow_missing_dataset=True) option")
    def random_message(self):
        """
        Generate a random message for watermarking.

        Returns:
            torch.Tensor: Random message tensor.
        """
        raise NotImplementedError

    def build_models(self):
        """
        Build and initialize models and optimizers as specified in the configuration.
        """
        raise NotImplementedError

    def load_models(self, checkpoint):
        """
        Load model and optimizer checkpoints from a directory.

        Args:
            checkpoint (str or Path): Path to the checkpoint directory.
        """
        raise NotImplementedError

    def eval(self):
        """
        Evaluate the model on the validation or test set.
        """
        raise NotImplementedError

    def eval_mode(self):
        """
        Set the solver and model to evaluation mode.
        """
        logger.debug("eval mode")
        self.mode  ='test'
                            
    def init_visqol(self):
        """Initialize VisQOL [1] API for perceptual metrics computation.

        [1] A. Hines, J. Skoglund, A. C. Kokaram, and N. Harte, “ViSQOL: An objective speech quality
            model,” EURASIP J. Audio, Speech, Music Process., vol. 2015, pp. 1–18, 2015.
        """
        from visqol import visqol_lib_py
        from visqol.pb2 import visqol_config_pb2

        config_16k = visqol_config_pb2.VisqolConfig()
        config_16k.audio.sample_rate = 16000
        config_16k.options.use_speech_scoring = True
        config_16k.options.svr_model_path = os.path.join(
            os.path.dirname(visqol_lib_py.__file__), "model", 
            "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite")
        self.visqol_api_16k = visqol_lib_py.VisqolApi()
        self.visqol_api_16k.Create(config_16k)

        config_48k = visqol_config_pb2.VisqolConfig()
        config_48k.audio.sample_rate = 48000
        config_48k.options.use_speech_scoring = False
        config_48k.options.svr_model_path = os.path.join(
            os.path.dirname(visqol_lib_py.__file__), "model", "libsvm_nu_svr_model.txt")
        self.visqol_api_48k = visqol_lib_py.VisqolApi()
        self.visqol_api_48k.Create(config_48k)

    def compute_perceptual_metrics(self,
                                   audio_filepath: str,
                                   start_time: float,
                                   audio_duration: float,
                                   watermarked_audio: torch.Tensor,
                                   distorted_audio: torch.Tensor) -> Dict[str, float]:
        """Compute perceptual metrics using MCD (Mel Cepstral Distance [1] and VisQOL [2].

        [1] J. Sternkopf and S. Taubert, mel-cepstral-distance (Version 0.0.3) [Computer software],
            2024. [Online]. Available: https://doi.org/10.5281/zenodo.10567255 
        [2] A. Hines, J. Skoglund, A. C. Kokaram, and N. Harte, “ViSQOL: An objective speech quality
            model,” EURASIP J. Audio, Speech, Music Process., vol. 2015, pp. 1–18, 2015.

        Args:
            audio_filepath (str): Path to the original audio file.
            start_time (float): Start time of the segment.
            audio_duration (float): Duration of the segment.
            watermarked_audio (torch.Tensor): Watermarked audio tensor.
            distorted_audio (torch.Tensor): Distorted audio tensor.

        Returns:
            Dict[str, float]: Dictionary with the following keys:
            - mcd_wm_48k: MCD between the original audio and the watermarked audio (48 kHz).
            - mcd_wm_16k: MCD between the original audio and the watermarked audio (16 kHz).
            - mcd_distorted_48k: MCD between the original audio and the distorted audio (48 kHz).
            - mcd_distorted_16k: MCD between the original audio and the distorted audio (16 kHz).
            - moslqo_wm_48k: MOS-LQO between the original audio and the watermarked audio (48 kHz).
            - moslqo_wm_16k: MOS-LQO between the original audio and the watermarked audio (16 kHz).
            - moslqo_distorted_48k: MOS-LQO between the original audio and the distorted audio (48 kHz).
            - moslqo_distorted_16k: MOS-LQO between the original audio and the distorted audio (16 kHz).        
        """
        metadata = torchaudio.info(audio_filepath)
        orig_sample_rate = metadata.sample_rate
        audio_orig, _ = torchaudio.load(audio_filepath, 
                                        int(start_time*orig_sample_rate), 
                                        num_frames=int(audio_duration*orig_sample_rate))

        y_wm = watermarked_audio.detach().to('cpu')
        y_distorted = distorted_audio.detach().to('cpu')
        if orig_sample_rate != 16000:
            y_16k = torchaudio.transforms.Resample(orig_freq=orig_sample_rate,
                                                   new_freq=16000)(audio_orig)
        else:
            y_16k = audio_orig.clone()

        if self.sample_rate != 16000:
            y_distorted_16k = torchaudio.transforms.Resample(orig_freq=self.sample_rate, 
                                                             new_freq=16000)(y_distorted)
            y_wm_16k = torchaudio.transforms.Resample(orig_freq=self.sample_rate, 
                                                      new_freq=16000)(y_wm)
        else:
            y_distorted_16k = y_distorted.clone()
            y_wm_16k = y_wm.clone()


        # In-memory paths for the audio files instead of writing to disk
        y_16k_path = io.BytesIO()
        y_distorted_16k_path = io.BytesIO()
        y_wm_16k_path = io.BytesIO()
 
        # compare_audio_files reads from the disk, so we need to save the audio files first.
        # This is not optimal, but let's keep it for now.
        sf.write(y_16k_path, y_16k.view(-1), 16000, format='wav')
        sf.write(y_distorted_16k_path, y_distorted_16k.view(-1), 16000, format='wav')
        sf.write(y_wm_16k_path, y_wm_16k.view(-1), 16000, format='wav')
        
        [buffer.seek(0) for buffer in [y_16k_path,
                                       y_distorted_16k_path,
                                       y_wm_16k_path]]
        
        mcd_wm_16k, _ = compare_audio_files(y_16k_path, y_wm_16k_path)
        mcd_distorted_16k, _ = compare_audio_files(y_16k_path, y_distorted_16k_path)

        [buffer.close() for buffer in [y_16k_path,
                                       y_distorted_16k_path,
                                       y_wm_16k_path]]
        
        moslqo_wm_16k = self.visqol_api_16k.Measure(y_16k.view(-1).to(torch.float64).numpy(), 
                                                    y_wm_16k.view(-1).to(torch.float64).numpy()).moslqo
        moslqo_distorted_16k = self.visqol_api_16k.Measure(y_16k.view(-1).to(torch.float64).numpy(), 
                                                           y_distorted_16k.view(-1).to(torch.float64).numpy()).moslqo
        
        return {
            'mcd_wm_16k': mcd_wm_16k,
            'mcd_distorted_16k': mcd_distorted_16k,
            'moslqo_wm_16k': moslqo_wm_16k,
            'moslqo_distorted_16k': moslqo_distorted_16k        
        }

    def seed_everything(self, seed: int):
        """
        Set random seeds for reproducibility.

        Args:
            seed (int): Random seed value.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

    def aggregate_per_track2(self, 
                             data_frame, 
                             sum_columns):
        """
        Aggregate results per audio track, summing and averaging specified columns.

        Args:
            data_frame (pd.DataFrame): DataFrame with results.
            sum_columns (list): Columns to sum (others are averaged).

        Returns:
            pd.DataFrame: Aggregated DataFrame per track.
        """
        df_per_track = data_frame.copy()

        grouped_by_track = df_per_track.groupby(['audio_filepath', 'attack_type', 'attack_params'])

        columns_to_aggregate = set(df_per_track.columns) - {'audio_filepath', 'dataset', 'attack_type', 'attack_params', 'chunk_index', 'num_frames'}
        # hard_columns = [col for col in columns_to_aggregate if col.startswith('hard')]
        mean_columns = [col for col in columns_to_aggregate if col not in sum_columns]

        # Aggregate the columns
        df_per_track['num_frames'] = grouped_by_track['num_frames'].transform('sum')
        df_per_track[sum_columns] = grouped_by_track[sum_columns].transform('sum')
        df_per_track[mean_columns] = grouped_by_track[mean_columns].transform('mean')

        for sum_column in sum_columns:
            df_per_track[sum_column] = df_per_track[sum_column] / df_per_track['num_frames']
            df_per_track[sum_column.replace('num_correct', 'bitwise')] = df_per_track[sum_column] / self.config.nbits

        # Drop duplicate rows to get a full view of the grouped data
        df_per_track = df_per_track.drop_duplicates(subset=['audio_filepath', 'dataset', 'attack_type', 'attack_params'])
        return df_per_track
        
    def aggregate_per_track(self, 
                            dataframe):
        """
        Aggregate results per audio track, averaging and multiplying specified columns.

        Args:
            dataframe (pd.DataFrame): DataFrame with results.

        Returns:
            pd.DataFrame: Aggregated DataFrame per track.
        """
        df_per_track = dataframe.copy()
        # Group rows by the second column (track name)
        grouped_by_track = df_per_track.groupby('audio_filepath')
        # Log warning if 'dataset', 'attack_type', and 'attack_params' have different values for each group
        for name, group in grouped_by_track:
            for column in ['dataset', 'attack_type', 'attack_params']:
                if group[column].nunique() != 1:
                    logger.warning(f"Different {column} values found in group {name}")
        
        # Set of columns except for audio_filepath, dataset, attack_type, attack_params, chunk_index
        columns_to_aggregate = set(df_per_track.columns) - {'audio_filepath', 'dataset', 'attack_type', 'attack_params', 'chunk_index'}
        hard_columns = [col for col in columns_to_aggregate if col.startswith('hard')]
        very_hard_columns = [f'very_{col}' for col in columns_to_aggregate if col.startswith('hard')]
        mean_columns = [col for col in columns_to_aggregate if col not in hard_columns]

        # Aggregate the columns
        df_per_track[hard_columns] = grouped_by_track[hard_columns].transform('mean')
        df_per_track[very_hard_columns] = grouped_by_track[hard_columns].transform('prod')
        df_per_track[mean_columns] = grouped_by_track[mean_columns].transform('mean')

        # Drop duplicate rows to get a full view of the grouped data
        df_per_track = df_per_track.drop_duplicates(subset=['audio_filepath', 'dataset', 'attack_type', 'attack_params'])
        return df_per_track
        
    def compute_agg(self, df_result, metrics, csv_suffix, key_columns, prefix):
        """
        Compute and save mean results grouped by attack type, dataset, and (dataset, attack type).

        Args:
            df_result (pd.DataFrame): DataFrame with results.
            metrics (list): List of metric columns.
            csv_suffix (str): Suffix for output CSV files.
            key_columns (list): Key columns to include in the output.
            prefix (str): Prefix for logging and file naming.
        """
        save_path_1 = os.path.join(self.test_results_dir, f'{prefix}_mean_by_attack{csv_suffix}.csv')
        mean_by_attack = compute_mean_by_group('attack_type', df_result, metrics, key_columns)
        mean_by_attack.to_csv(save_path_1, 
                              sep=self.csv_delimiter,
                              index=False)
        logger.info(f"{prefix}: Mean results grouped by attack type was saved at {save_path_1}")
        
        self.exp_logger.log_wandb_if_possible(
            {f"{prefix}/group_by_attack": wandb.Table(dataframe=mean_by_attack)})
            
        # Compute mean of specified columns grouped by dataset
        save_path_2=os.path.join(self.test_results_dir, f'{prefix}_mean_by_dataset{csv_suffix}.csv')        
        mean_by_dataset = compute_mean_by_group('dataset', df_result, metrics, key_columns)
        mean_by_dataset.to_csv(save_path_2, 
                               sep=self.csv_delimiter,
                               index=False)
        logger.info(f"{prefix}: Mean results grouped by dataset was saved at {save_path_2}")
        
        self.exp_logger.log_wandb_if_possible(
            {f"{prefix}/group_by_dataset": wandb.Table(dataframe=mean_by_dataset)})
        
        # Compute mean of specified columns grouped by (dataset, attack type)
        save_path_3 = os.path.join(self.test_results_dir, f'{prefix}_mean_by_dataset_attack{csv_suffix}.csv')
        mean_by_dataset_attack = compute_mean_by_group(['dataset', 'attack_type'], df_result, metrics, key_columns)
        mean_by_dataset_attack.to_csv(save_path_3, 
                                      sep=self.csv_delimiter,
                                      index=False)
        logger.info(f"{prefix}: Mean results grouped by (dataset, attack type) was saved at {save_path_3}")

        self.exp_logger.log_wandb_if_possible(
            {f"{prefix}/group_by_dataset_attack": wandb.Table(dataframe=mean_by_dataset_attack)})
        
        if prefix == 'chunklv':
            logger.info("summary of results per attack type:")
            for _, row in mean_by_attack.iterrows():
                logger.info(
                    f"-type: {row['attack_type']}, "
                    f"bitwise/distorted: {row.get('bitwise/distorted', 'N/A'):.2f}, "
                    f"hard/distorted: {row.get('hard/distorted', 'N/A'):.2f}"
                )
