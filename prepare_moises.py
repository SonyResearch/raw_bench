import argparse
import glob
import librosa
import numpy as np
import os
import pandas as pd
import soundfile as sf
from loguru import logger
from tqdm import tqdm

from download_datasets import dataset_process_decorator

def main(moisesdb_dir):

    global MOISES_DIR # Make moisesdb_dir global to use in process_moisesdb
    MOISES_DIR = moisesdb_dir
    TEST_DIR = 'test_data'
    
    process_moisesdb(os.path.join(TEST_DIR, "MoisesDB"), None, None)
    
    
@dataset_process_decorator
def process_moisesdb(moisesdb_dir, tmp_dir, rm_tmp):
    """
    Process MoisesDB dataset.
    """
    sr = 44100

    os.makedirs(moisesdb_dir, exist_ok=True)
    logger.info("Processing MoisesDB dataset...")

    # Note that test_loose.csv also has the same entries for MoisesDB, so we process test_strict.csv only.
    df_test = pd.read_csv('data/test_strict.csv', sep='|')

    for audio_filepath in tqdm(df_test[df_test['dataset_name']=='MoisesDB']['audio_filepath'].unique()):
        song_name = audio_filepath[:-4]
        wav_files = glob.glob(os.path.join(MOISES_DIR, song_name, '**', '*.wav'), recursive=True)
        # Load all wavs
        wav_arrays = []
        min_length = None
        for path in wav_files:
            wav, _ = librosa.load(path, sr=sr, mono=True)
            if min_length is None or len(wav) < min_length:
                min_length = len(wav)
            wav_arrays.append(wav)

        # Truncate and mix
        wav_arrays = [wav[:min_length] for wav in wav_arrays]
        mixed = np.sum(wav_arrays, axis=0)

        max_val = np.max(np.abs(mixed))
        if max_val > 1:
            # actually there is no clipping when we tested
            logger.warning(f"Max value in mixed audio exceeds 1, normalizing: {max_val}")

        output_filepath = os.path.join(moisesdb_dir, song_name + '.wav')
        sf.write(output_filepath, mixed.T, sr)    
    
    logger.info("MoisesDB processing completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process datasets.")
    parser.add_argument('--moisesdb-dir', type=str, required=True, help='path of moisesdb-v0.1 dataset')
    args = parser.parse_args()
    main(args.moisesdb_dir)
