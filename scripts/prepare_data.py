"""
Data Preprocessing Pipeline
---------------------------
1. Load the training data from the directory specified in SETTINGS.json.
2. Run any preprocessing or synthetic data generation steps.
3. Save or return the final cleaned data.
"""

import json
import logging
import argparse
import pandas as pd
from pathlib import Path
from src.preprocessing.preprocess import add_subject
from chain_of_thought.generate_cot import infer_cot
from synthetic_data_generation.generation1 import generate_1
from synthetic_data_generation.generation2 import generate_2
from synthetic_data_generation.generation3 import generate_3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Data preprocessing pipeline")
    parser.add_argument(
        "--skip_render",
        action="store_true",
        help="Skip synthetic data generation and load from existing files"
    )
    return parser.parse_args()

def load_settings(settings_file="SETTING.json"):
    """
    Load settings from a JSON file.
    Raises:
        FileNotFoundError: If the settings file does not exist.
        json.JSONDecodeError: If the JSON is not properly formatted.
    """
    try:
        with open(settings_file, "r") as file:
            settings = json.load(file)
    except FileNotFoundError as e:
        logging.error(f"Settings file not found at: {settings_file}")
        raise e
    except json.JSONDecodeError as e:
        logging.error("Invalid JSON format in settings file.")
        raise e
    
    return settings

def load_and_prepare_raw_training_data(raw_data_dir: Path) -> pd.DataFrame:
    """
    Load raw training data from the specified directory and apply basic transformations.
    
    Args:
        raw_data_dir (Path): Directory that contains the raw 'train.csv' file.
    
    Returns:
        pd.DataFrame: Raw training data with initial transformations applied.
    """
    train_path = raw_data_dir / "train.csv"
    logging.info(f"Loading raw training data from: {train_path}")
    
    try:
        train = pd.read_csv(train_path)
    except FileNotFoundError as e:
        logging.error(f"Train file not found at {train_path}")
        raise e
    
    # Example transformation: add a constant value column
    # train["quality-gpt4o-mini"] = 5
    
    # You could add fold-splitting logic here if needed
    return train

def apply_cot_inference(target: pd.DataFrame) -> pd.DataFrame:
    """
    Apply Chain of Thought (CoT) inference to the target DataFrame.
    
    Args:
        target (pd.DataFrame): DataFrame to which CoT inference will be applied.
    
    Returns:
        pd.DataFrame: DataFrame after CoT inference.
    """
    return infer_cot(target)

def attach_subject_information(target: pd.DataFrame) -> pd.DataFrame:
    """
    Add subject information to the target DataFrame.
    
    Args:
        target (pd.DataFrame): DataFrame to which subject information will be added.
    
    Returns:
        pd.DataFrame: DataFrame with appended subject information.
    """
    return add_subject(target)

def preprocess_data(skip_render: bool) -> pd.DataFrame:
    """
    Preprocess data according to the pipeline.
    If 'skip_render' is True, read pre-rendered synthetic data.
    Otherwise, generate and process synthetic data from scratch.
    
    Args:
        skip_render (bool): Whether to skip rendering synthetic data and load from existing files.
    Returns:
        pd.DataFrame: Final concatenated and validated dataset.
    """
    settings = load_settings()
    
    # Extract paths from settings and convert them to Path objects
    raw_data_dir = Path(settings["RAW_DATA_DIR"])
    synthetic_dir = Path(settings["SYNTHETIC_DATA_DIR"])
    clean_train_path = Path(settings["TRAIN_DATA_CLEAN_PATH"])
    
    if skip_render:
        # Load the existing processed data
        logging.info("Skipping synthetic data generation. Loading existing files.")
        try:
            train = pd.read_csv(clean_train_path)
            g1 = pd.read_csv(f"{synthetic_dir}/synthetic-round1-render.csv")
            g2 = pd.read_csv(f"{synthetic_dir}/synthetic-round2-render.csv")
            g3 = pd.read_csv(f"{synthetic_dir}/synthetic-round3-render.csv")
        except FileNotFoundError as e:
            logging.error(f"One of the required data files was not found: {e}")
            raise e
    else:
        # Generate fresh data from the raw training data
        train = load_and_prepare_raw_training_data(raw_data_dir)
        
        logging.info("Generating synthetic data: round 1")
        g1 = generate_1(synthetic_dir)
        
        logging.info("Generating synthetic data: round 2")
        g2 = generate_2(synthetic_dir, g1)
        
        logging.info("Generating synthetic data: round 3")
        g3 = generate_3(synthetic_dir, g1, g2)
        
        # Apply CoT inference and subject labeling to each synthetic dataset
        logging.info("Applying CoT inference and subject information to synthetic data.")
        g1, g2, g3 = [
            attach_subject_information(apply_cot_inference(g)) 
            for g in (g1, g2, g3)
        ]
    
    # Define required columns
    required_cols = [
        'MisconceptionId', 'MisconceptionName', 'QuestionText',
        'AnswerText', 'CorrectAnswerText', 'quality-gpt4o-mini', 'SubjectName',
        'FirstSubjectId', 'FirstSubjectName', 'SecondSubjectId', 'SecondSubjectName',
        'ThirdSubjectId', 'ThirdSubjectName', 
        'p000-qwen25-32b-instruct-cot_misunderstanding', 'SubjectId',
        'QuestionId', 'QuestionId_Answer', 'fold'
    ]
    
    # Concatenate the main training data with the synthetic data
    logging.info("Concatenating all datasets...")
    final_data = pd.concat([train, g1, g2, g3], ignore_index=True)
    
    # --- Column Existence Check ---
    logging.info("Checking for missing required columns...")
    missing_cols = [col for col in required_cols if col not in final_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in the final dataset: {missing_cols}")
    
    # --- Missing Values Check ---
    logging.info("Checking for missing values in required columns...")
    missing_val_mask = final_data[required_cols].isnull()
    if missing_val_mask.any().any():
        # We can gather which columns have missing values
        cols_with_missing = missing_val_mask.any(axis=0)
        cols_with_missing = list(cols_with_missing[cols_with_missing].index)
        raise ValueError(f"Missing values detected in columns: {cols_with_missing}")
    
    logging.info("Final dataset validation passed. No missing columns or values.")
    return final_data

if __name__ == "__main__":
    args = parse_arguments()
    processed_data = preprocess_data(args.skip_render)
    # You can save the final data if desired, e.g.:
    processed_data.to_csv("data/clean/final_dataset.csv", index=False)
    logging.info("Data preprocessing pipeline completed successfully.")
