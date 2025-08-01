import argparse
import os
import pandas as pd
import py7zr
import shutil
import subprocess
import tarfile
import urllib.request
import zipfile

from loguru import logger
from tqdm import tqdm
import huggingface_hub

from raw_bench.utils import file_checksum

def main(tmp_dir, rm_tmp):

    TEST_DIR = 'test_data'
    os.makedirs(TEST_DIR, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # List of dataset processing functions and their target subdirectories
    datasets = [
        (process_jaCappella, 'jaCappella'),
        (process_bach10, 'Bach10'),
        (process_clotho, 'Clotho'),
        (process_air, 'AIR'),
        (process_daps, 'DAPS'),
        (process_freischuetz, 'Freischuetz'),
        (process_guitarset, 'GuitarSet'),
        (process_maestro, 'MAESTRO'),
        (process_pcd, 'PCD'),
        (process_demand, 'DEMAND'),
    ]

    for func, subdir in datasets:
        func(os.path.join(TEST_DIR, subdir), tmp_dir, rm_tmp)

def dataset_process_decorator(process_func):
    def wrapper(dataset_dir, tmp_dir, rm_tmp):
        if is_dataset_processed(dataset_dir):
            logger.info(f"Dataset {dataset_dir} already processed. Skipping.")
            return
        result = process_func(dataset_dir, tmp_dir, rm_tmp)
        mark_done(dataset_dir)
        return result
    return wrapper

@dataset_process_decorator
def process_bach10(bach10_dir, tmp_dir, rm_tmp):
    """
    Download and process Bach10 Clotho dataset.

    Args:
        bach10_dir: str
            Target directory to store processed Bach10 audio files.
        tmp_dir: str
            Temporary directory for downloads and extraction.
        rm_tmp: bool
            Whether to remove temporary files after processing.
    """
    # Download the Bach10 Dataset from GitHub
    tmp_bach10_dir = os.path.join(tmp_dir, 'tmp_bach10')
    if os.path.exists(tmp_bach10_dir):
        shutil.rmtree(tmp_bach10_dir)
    subprocess.run(['git', 'clone', 'https://github.com/flippy-fyp/Bach10_v1.1', tmp_bach10_dir], check=True)

    os.makedirs(bach10_dir, exist_ok=True)

    for dir_name in tqdm(os.listdir(tmp_bach10_dir)):
        if not dir_name.startswith('0') and not dir_name.startswith('1'):
            continue
        subdir_path = os.path.join(tmp_bach10_dir, dir_name)        

        # copy the representative wav file for each directory
        wav_file_path = os.path.join(subdir_path, f"{dir_name}.wav")        
        if os.path.isfile(wav_file_path):
            shutil.move(wav_file_path, os.path.join(bach10_dir, f"{dir_name}.wav"))
        else:
            raise FileNotFoundError(f"Expected .wav file not found: {wav_file_path}. Please check the repository")
            
    if rm_tmp:
        shutil.rmtree(tmp_bach10_dir)

def mark_done(BACH10_DIR):
    with open(os.path.join(BACH10_DIR, '.done'), 'w') as f:
        pass

def is_dataset_processed(dataset_dir):
    return os.path.exists(os.path.join(dataset_dir, '.done'))

@dataset_process_decorator
def process_clotho(clotho_dir, tmp_dir, rm_tmp):
    """
    Download and process the Clotho dataset.

    Args:
        clotho_dir: str
            Target directory to store processed Clotho audio files.
        tmp_dir: str
            Temporary directory for downloads and extraction.
        rm_tmp: bool
            Whether to remove temporary files after processing.
    """
    # Download the Clotho dataset
    url = "https://zenodo.org/records/4783391/files/clotho_audio_evaluation.7z"
    clotho_7z_path = os.path.join(tmp_dir, "clotho_evaluation.7z")
    skip_download = False
    if os.path.exists(clotho_7z_path):
        try:
            with py7zr.SevenZipFile(clotho_7z_path, mode='r') as archive:
                archive.list()  # Try listing contents to check validity
            logger.info(f"{clotho_7z_path} already exists and is a valid .7z file. Skipping download.")
            skip_download = True
        except Exception:
            logger.warning(f"{clotho_7z_path} exists but is not a valid .7z file. Re-downloading...")
            os.remove(clotho_7z_path)
    else:
        logger.info(f"{clotho_7z_path} does not exist. Proceeding to download.")

    if not skip_download:
        logger.info(f"Downloading Clotho dataset from {url} ...")
        try:
            subprocess.run(['wget', '-O', clotho_7z_path, url], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("'wget' not found or failed, falling back to urllib.request...")
            urllib.request.urlretrieve(url, clotho_7z_path)
        logger.info("Download complete.")

    # Extract archive into a temporary directory
    clotho_unzip_path = os.path.join(tmp_dir, 'tmp_clotho')
    os.makedirs(clotho_unzip_path, exist_ok=True)

    with py7zr.SevenZipFile(clotho_7z_path, mode='r') as archive:
        archive.extractall(path=clotho_unzip_path)

    # Copy .wav files to target directory, removing whitespace from filenames
    os.makedirs(clotho_dir, exist_ok=True)

    for fname in tqdm(os.listdir(os.path.join(clotho_unzip_path, 'evaluation'))):
        if not fname.lower().endswith('.wav'):
            continue
        # Remove white spaces within the filenames
        cleaned_name = fname.replace(' ', '')
        shutil.move(os.path.join(clotho_unzip_path, 'evaluation', fname), 
                    os.path.join(clotho_dir, cleaned_name))

    # Clean up
    shutil.rmtree(clotho_unzip_path)
    if rm_tmp:
        os.remove(clotho_7z_path)

    logger.info(f"Clotho dataset ready in '{clotho_dir}'")   

@dataset_process_decorator
def process_air(air_dir, tmp_dir, rm_tmp):
    """
    Download and process the AIR dataset.

    Args:
        air_dir: str
            Target directory to store processed AIR audio files.
        tmp_dir: str
            Temporary directory for downloads and extraction.
        rm_tmp: bool
            Whether to remove temporary files after processing.
    """
    os.makedirs(air_dir, exist_ok=True)

    # Download ZIP
    air_url = "https://www.iks.rwth-aachen.de/fileadmin/user_upload/downloads/forschung/tools-downloads/air_database_release_1_4.zip"
    air_zip = os.path.join(tmp_dir, "air_database_release_1_4.zip")
    air_tmp = os.path.join(tmp_dir, "AIR_tmp")

    if os.path.exists(air_zip):
        try:
            # Try to open as a zip file to check validity
            with zipfile.ZipFile(air_zip, 'r') as zf:
                bad_file = zf.testzip()
                if bad_file is not None:
                    raise zipfile.BadZipFile(f"Corrupted file: {bad_file}")
            logger.info(f"{air_zip} already exists and is a valid zip file. Skipping download.")
        except Exception:
            logger.warning(f"{air_zip} exists but is not a valid zip file. Re-downloading...")
            os.remove(air_zip)
    if not os.path.exists(air_zip):
        logger.info("Downloading AIR dataset...")
        try:
            subprocess.run(['wget', air_url, '-O', air_zip], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("'wget' not found or failed, falling back to urllib.request...")
            urllib.request.urlretrieve(air_url, air_zip)

    # Unzip main archive to temp folder
    logger.info("Extracting main archive...")
    os.makedirs(air_tmp, exist_ok=True)
    if os.path.exists(air_tmp):
        shutil.rmtree(air_tmp)
        
    subprocess.run(['unzip', '-q', air_zip, '-d', air_tmp], check=True)

    # Move inner ZIP to working directory
    inner_zip_path = os.path.join(air_tmp, "AIR_1_4", "AIR_wav_files.zip")
    shutil.move(inner_zip_path, ".")

    # Clean up temp extraction
    shutil.rmtree(air_tmp)

    # Extract AIR_wav_files.zip
    logger.info("Extracting audio files...")
    subprocess.run(['unzip', '-q', 'AIR_wav_files.zip', '-d', air_dir], check=True)

    # Remove leftover ZIPs
    os.remove("AIR_wav_files.zip")

    if rm_tmp:
        os.remove(air_zip)

    logger.info(f"AIR dataset ready in {air_dir}")

@dataset_process_decorator
def process_daps(daps_dir, tmp_dir, rm_tmp):
    """
    Download and process the DAPS dataset.

    Args:
        daps_dir: str
            Target directory to store processed DAPS audio files.
        tmp_dir: str
            Temporary directory for downloads and extraction.
        rm_tmp: bool
            Whether to remove temporary files after processing.
    """

    # Download DAPS tarball
    url = "https://zenodo.org/records/4660670/files/daps.tar.gz?download=1"
    daps_tar_path = os.path.join(tmp_dir, "daps.tar.gz")
    daps_unzip_path = os.path.join(tmp_dir, "DAPS_tmp")
    os.makedirs(daps_unzip_path, exist_ok=True)

    download_required = False
    extract_required = False
    if os.path.exists(daps_tar_path):
        try:
            logger.info(f"{daps_tar_path} already exists. Skipping download.")
            # Extract archive into a temporary directory
            logger.info("Extracting DAPS tarball...")
            with tarfile.open(daps_tar_path, "r:gz") as tar:
                tar.extractall(path=daps_unzip_path)        
            download_required = False
            extract_required = False
        except:
            logger.warning(f"{daps_tar_path} exists but extraction failed. Re-downloading...")
            download_required = True
            extract_required = True
    
    if download_required:
        logger.info(f"Downloading DAPS dataset from {url} ...")
        try:
            subprocess.run(['wget', '-O', daps_tar_path, url], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("'wget' not found or failed, falling back to urllib.request...")
            urllib.request.urlretrieve(url, daps_tar_path)
        logger.info("Download complete.")
    
    if extract_required:
        logger.info("Extracting DAPS tarball...")
        with tarfile.open(daps_tar_path, "r:gz") as tar:
            tar.extractall(path=daps_unzip_path)            

    # Prepare output directory
    os.makedirs(daps_dir, exist_ok=True)

    # Load test CSV to get file mappings
    df_test = pd.read_csv('data/test_strict.csv', sep='|')
    for idx, row in df_test[df_test['dataset_name'] == 'DAPS'].iterrows():
        orig_filepath = row['orig_filepath']
        audio_filepath = row['audio_filepath']
        src = os.path.join(daps_unzip_path, 'daps', orig_filepath)
        dst = os.path.join(daps_dir, audio_filepath)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)

    # Clean up
    shutil.rmtree(daps_unzip_path)
    if rm_tmp:
        os.remove(daps_tar_path)

    logger.info(f"DAPS dataset ready in '{daps_dir}'")

@dataset_process_decorator
def process_jaCappella(jacappella_dir, tmp_dir, rm_tmp):
    # Download jaCappella dataset using huggingface_hub
    tmp_jacappella_dir = os.path.join(tmp_dir, "jaCappella_tmp")
    huggingface_hub.snapshot_download(
        repo_id="jaCappella/jaCappella",
        repo_type="dataset",
        local_dir=os.path.join(tmp_jacappella_dir, "jaCappella"),
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    os.makedirs(jacappella_dir, exist_ok=True)

    df_test = pd.read_csv('data/test_strict.csv', sep='|')
    for idx, row in df_test[df_test['dataset_name'] == 'jaCappella'].iterrows():
        rel_path = row['audio_filepath']
        orig_filepath = row['orig_filepath']
        shutil.copy(os.path.join(tmp_jacappella_dir, orig_filepath), os.path.join(jacappella_dir, rel_path))

    if rm_tmp:
        shutil.rmtree(tmp_jacappella_dir)

    logger.info(f"jaCappella dataset ready in '{jacappella_dir}'")

@dataset_process_decorator
def process_freischuetz(freischuetz_dir, tmp_dir, rm_tmp):
    """
    Download and process the Freischütz dataset based on Freischuetz.txt.

    Args:
        freischuetz_dir: str
            Target directory to store downloaded Freischütz audio files.
        tmp_dir: str
            Temporary directory for downloads (not used here, but kept for interface consistency).
        rm_tmp: bool
            Whether to remove temporary files after processing (not used here).
    """
    os.makedirs(freischuetz_dir, exist_ok=True)

    txt_path = 'data/Freischuetz.txt'
    hash_path = 'data/Freischuetz-hash.txt'

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"{txt_path} not found. Please provide the Freischuetz.txt file with URLs.")    
    if not os.path.exists(hash_path):
        raise FileNotFoundError(f"{hash_path} not found. Please provide the Freischuetz-hash.txt file with checksums.")

    with open(hash_path, 'r') as f:
        hash_dict = {line.split('\t')[0]: line.strip().split('\t')[1] for line in f if line.strip()}

    with open(txt_path, 'r') as f:
        urls = [line.strip() for line in f if line.strip()]

    for url in urls:
        filename = url.split('/')[-1].replace(' ', '')
        out_path = os.path.join(freischuetz_dir, filename)

        if not os.path.exists(out_path):
            logger.info(f"Downloading {filename} ...")
            try:
                subprocess.run(['wget', '-q', '-O', out_path, url], check=True)
            except (FileNotFoundError, subprocess.CalledProcessError):
                logger.warning(f"'wget' not found or failed for {url}, falling back to urllib.request...")
                import urllib.request
                urllib.request.urlretrieve(url, out_path)
        
        # check checksum
        if hash_dict[filename] != file_checksum(out_path):
            logger.warning(f"Checksum of {filename} is incorrect. Expected {hash_dict[filename]}, got {file_checksum(out_path)}")

    logger.info(f"Freischuetz dataset ready in '{freischuetz_dir}'")

@dataset_process_decorator
def process_guitarset(guitarset_dir, tmp_dir, rm_tmp):
    """
    Download and process the GuitarSet (audio_mono-mic subset) dataset.

    Args:
        guitarset_dir: str
            Target directory to store processed GuitarSet audio files.
        tmp_dir: str
            Temporary directory for downloads and extraction.
        rm_tmp: bool
            Whether to remove temporary files after processing.
    """
    # Download the audio_mono-mic subset of GuitarSet
    url = "https://zenodo.org/records/3371780/files/audio_mono-mic.zip?download=1"
    zip_path = os.path.join(tmp_dir, "audio_mono-mic.zip")
    unzip_path = os.path.join(tmp_dir, "GuitarSet_tmp")
    os.makedirs(unzip_path, exist_ok=True)

    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                bad_file = zf.testzip()
                if bad_file is not None:
                    raise zipfile.BadZipFile(f"Corrupted file: {bad_file}")
            logger.info(f"{zip_path} already exists and is a valid zip file. Skipping download.")
        except Exception:
            logger.warning(f"{zip_path} exists but is not a valid zip file. Re-downloading...")
            os.remove(zip_path)
    if not os.path.exists(zip_path):
        logger.info(f"Downloading GuitarSet dataset from {url} ...")
        try:
            subprocess.run(['wget', '-O', zip_path, url], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("'wget' not found or failed, falling back to urllib.request...")
            urllib.request.urlretrieve(url, zip_path)
        logger.info("Download complete.")

    # Extract archive into a temporary directory
    logger.info("Extracting GuitarSet zip...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(unzip_path)

    # Prepare output directory
    os.makedirs(guitarset_dir, exist_ok=True)

    # Load test CSV to get file mappings
    df_test = pd.read_csv('data/test_strict.csv', sep='|')
    relevant_files = df_test[df_test['dataset_name'] == 'GuitarSet']['audio_filepath'].tolist()
    for rel_path in relevant_files:
        src = os.path.join(unzip_path, rel_path)
        dst = os.path.join(guitarset_dir, rel_path)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)

    # Clean up
    shutil.rmtree(unzip_path)
    if rm_tmp:
        os.remove(zip_path)

    logger.info(f"GuitarSet dataset ready in '{guitarset_dir}'")    

@dataset_process_decorator
def process_maestro(maestro_dir, tmp_dir, rm_tmp):
    """
    Download and process the MAESTRO v3.0.0 dataset.

    Args:
        maestro_dir: str
            Target directory to store processed MAESTRO audio files.
        tmp_dir: str
            Temporary directory for downloads and extraction.
        rm_tmp: bool
            Whether to remove temporary files after processing.
    """
    # Download MAESTRO v3.0.0
    url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
    zip_path = os.path.join(tmp_dir, "maestro-v3.0.0.zip")
    unzip_path = os.path.join(tmp_dir, "maestro_tmp")
    os.makedirs(unzip_path, exist_ok=True)

    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                logger.info(f"Extracting contents of {zip_path}")
                zf.extractall(unzip_path)
            logger.info(f"{zip_path} already exists and extracted successfully. Skipping download.")
            extraction_success = True
        except Exception as e:
            logger.warning(f"{zip_path} exists but extraction failed ({e}). Re-downloading...")
            os.remove(zip_path)
            extraction_success = False
    else:
        extraction_success = False
        
    if not extraction_success:
        logger.info(f"Downloading dataset from {url} ...")
        try:
            subprocess.run(['wget', '-O', zip_path, url], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("'wget' not found or failed, falling back to urllib.request...")
            urllib.request.urlretrieve(url, zip_path)
        logger.info("Download complete.")
        # Try extracting again
        with zipfile.ZipFile(zip_path, 'r') as zf:
            logger.info(f"Extracting contents of {zip_path}")            
            zf.extractall(unzip_path)

    # Prepare output directory
    os.makedirs(maestro_dir, exist_ok=True)

    # Load test CSV to get file mappings
    df_test = pd.read_csv('data/test_strict.csv', sep='|')
    for idx, row in df_test[df_test['dataset_name'] == 'MAESTRO'].iterrows():
        orig_filepath = row['orig_filepath']
        audio_filepath = row['audio_filepath']
        src = os.path.join(unzip_path, 'maestro-v3.0.0', orig_filepath)
        dst = os.path.join(maestro_dir, audio_filepath)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)

    # Clean up
    shutil.rmtree(unzip_path)
    if rm_tmp:
        os.remove(zip_path)

    logger.info(f"MAESTRO dataset ready in '{maestro_dir}'")
    
@dataset_process_decorator
def process_pcd(pcd_dir, tmp_dir, rm_tmp):
    """
    Download and process the PCD dataset.

    Args:
        pcd_dir: str
            Target directory to store processed PCD audio files.
        tmp_dir: str
            Temporary directory for downloads and extraction.
        rm_tmp: bool
            Whether to remove temporary files after processing.
    """
    import stat

    # Set up paths
    url = "https://www.audiolabs-erlangen.de/resources/MIR/PCD/PCD_2.0.1.zip"
    zip_path = os.path.join(tmp_dir, "PCD_2.0.1.zip")
    unzip_path = os.path.join(tmp_dir, "PCD_tmp")
    os.makedirs(unzip_path, exist_ok=True)

    # Download the zip file if not present or if extraction fails
    extraction_success = False
    if os.path.exists(zip_path):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(unzip_path)
            extraction_success = True
        except Exception as e:
            logger.warning(f"{zip_path} exists but extraction failed ({e}). Re-downloading...")
            os.remove(zip_path)
            extraction_success = False

    if not extraction_success:
        logger.info(f"Downloading PCD dataset from {url} ...")
        try:
            subprocess.run(['wget', '-O', zip_path, url], check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            logger.warning("'wget' not found or failed, falling back to urllib.request...")
            urllib.request.urlretrieve(url, zip_path)
        logger.info("Download complete.")
        # Try extracting again
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(unzip_path)

    # # Prepare output directory
    os.makedirs(pcd_dir, exist_ok=True)

    # Load test CSV to get file mappings
    df_test = pd.read_csv('data/test_strict.csv', sep='|')
    relevant_files = df_test[df_test["dataset_name"] == "PCD"]["file_id"].tolist()

    # Copy matching files
    logger.info(f"Moving {len(relevant_files)} matching excerpts...")
    excerpt_dir = os.path.join(unzip_path, "PCD_2.0.1", "excerpts")
    for file_id in relevant_files:
        rel_pc_id = file_id.replace("_P_reverb", "")
        source_path = os.path.join(excerpt_dir, rel_pc_id, f"{file_id}.wav")
        if os.path.exists(source_path):
            target_path = os.path.join(pcd_dir, f"{file_id}.wav")
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            shutil.move(source_path, target_path)
        else:
            logger.info(f"Warning: Missing {source_path}")

    # Clean up
    shutil.rmtree(unzip_path)
    if rm_tmp:
        os.remove(zip_path)

    logger.info(f"PCD dataset ready in '{pcd_dir}'")
        
@dataset_process_decorator
def process_demand(demand_dir, tmp_dir, rm_tmp):
    """
    Download and process the DEMAND dataset.

    Args:
        demand_dir: str
            Target directory to store processed DEMAND audio files.
        tmp_dir: str
            Temporary directory for downloads and extraction.
        rm_tmp: bool
            Whether to remove temporary files after processing.
    """
    os.makedirs(demand_dir, exist_ok=True)
    
    tmp_dir = os.path.join(tmp_dir, "tmp_DEMAND")
    os.makedirs(tmp_dir, exist_ok=True)
    
    locations = [
        "DKITCHEN", "DLIVING", "DWASHING", "NFIELD", "NPARK", "NRIVER",
        "OHALLWAY", "OMEETING", "OOFFICE", "PCAFETER", "PRESTO", "PSTATION",
        "SCAFE", "SPSQUARE", "STRAFFIC", "TBUS", "TCAR", "TMETRO"
    ]
    base_url = "https://zenodo.org/records/1227121/files"

    for loc in locations:
        zip_name = f"{loc}_48k.zip"
        url = f"{base_url}/{zip_name}?download=1"
        zip_path = os.path.join(tmp_dir, zip_name)

        skip_download = False
        if os.path.exists(zip_path):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    bad_file = zf.testzip()
                    if bad_file is not None:
                        raise zipfile.BadZipFile(f"Corrupted file: {bad_file}")
                logger.info(f"{zip_path} already exists and is a valid zip file. Skipping download.")
                skip_download = True
            except Exception:
                logger.warning(f"{zip_path} exists but is not a valid zip file. Re-downloading...")
                os.remove(zip_path)

        if not skip_download:
            logger.info(f"Downloading {zip_name}...")
            try:
                subprocess.run(['wget', '-O', zip_path, url], check=True)
            except (FileNotFoundError, subprocess.CalledProcessError):
                logger.warning(f"'wget' not found or failed for {url}, falling back to urllib.request...")
                urllib.request.urlretrieve(url, zip_path)

        logger.info(f"Extracting to {demand_dir}/{loc}_48k/...")
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(f"{demand_dir}/{loc}_48k")

        if rm_tmp:
            logger.info(f"Removing {zip_path}...")
            os.remove(zip_path)

    logger.info("All DEMAND files downloaded and extracted.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process datasets.")
    parser.add_argument('--tmp-dir', type=str, default='tmp', help='Temporary directory for downloads and extraction')
    parser.add_argument('--rm-tmp', action='store_true', help='Remove temporary files after processing')
    args = parser.parse_args()
    main(args.tmp_dir, args.rm_tmp)
