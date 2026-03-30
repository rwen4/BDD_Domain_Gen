import os
import pickle
import random
import re

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Time periods for wild time learning (3-year windows)
# 2008-2010, 2011-2013, 2014-2016, 2017-2019, 2020-2022
TIME_PERIODS = [2008, 2011, 2014, 2017, 2020]
TEST_SPLIT = 0.2


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def _resolve_data_file(data_dir, base_name):
    """Resolve path to data file, supporting both .csv and .csv.gz."""
    for ext in ('.csv', '.csv.gz'):
        path = os.path.join(data_dir, base_name + ext)
        if os.path.isfile(path):
            return path
    return None


def process_mimic_iv_data(data_dir):
    """
    Process MIMIC-IV data for mortality prediction.
    Expects: admissions.csv(.gz), patients.csv(.gz), diagnoses_icd.csv(.gz) in data_dir
    """
    set_seed(42)
    
    # Resolve required files (supports .csv and .csv.gz)
    file_map = {}
    for base in ['admissions', 'patients', 'diagnoses_icd']:
        path = _resolve_data_file(data_dir, base)
        if path is None:
            raise ValueError(f'Please provide {base}.csv or {base}.csv.gz in {data_dir}')
        file_map[base] = path
    
    # Load patients (include anchor_year_group for mapping shifted years to real years)
    patients = pd.read_csv(file_map['patients'])
    patient_cols = ['subject_id', 'gender', 'anchor_age', 'anchor_year']
    if 'anchor_year_group' in patients.columns:
        patient_cols.append('anchor_year_group')
    patients = patients[[c for c in patient_cols if c in patients.columns]].dropna(subset=['subject_id', 'anchor_year']).reset_index(drop=True)

    def _anchor_group_to_year(group_str):
        """Parse anchor_year_group (e.g. '2014 - 2016') to midpoint year."""
        if pd.isna(group_str):
            return np.nan
        m = re.search(r'(\d{4})\s*-\s*(\d{4})', str(group_str))
        if m:
            return int((int(m.group(1)) + int(m.group(2))) / 2)
        return np.nan

    if 'anchor_year_group' in patients.columns:
        patients['real_anchor_year'] = patients['anchor_year_group'].apply(_anchor_group_to_year)
    else:
        patients['real_anchor_year'] = np.nan
    
    # Load admissions (use 'race' if 'ethnicity' not present, per MIMIC-IV 2.x schema)
    admissions = pd.read_csv(file_map['admissions'])
    admissions['admittime'] = pd.to_datetime(admissions['admittime']).dt.date
    demo_col = 'ethnicity' if 'ethnicity' in admissions.columns else 'race'
    admissions = admissions[['subject_id', 'hadm_id', demo_col, 'admittime', 'hospital_expire_flag']]
    admissions = admissions.rename(columns={demo_col: 'ethnicity'})
    admissions = admissions.dropna()
    admissions['mortality'] = admissions['hospital_expire_flag'].astype(int)
    admissions = admissions.drop(columns=['hospital_expire_flag'])
    admissions = admissions.reset_index(drop=True)
    
    # Load diagnoses
    diagnoses = pd.read_csv(file_map['diagnoses_icd'])
    diagnoses = diagnoses.dropna()
    diagnoses = diagnoses.drop_duplicates()
    diagnoses = diagnoses.sort_values(by=['subject_id', 'hadm_id', 'seq_num'])
    
    # Convert to 3-digit ICD code (handle both string and numeric icd_code)
    diagnoses['icd_code_3digit'] = diagnoses['icd_code'].astype(str).str[:3]
    diagnoses_grouped = diagnoses.groupby(['subject_id', 'hadm_id'])['icd_code_3digit'].apply(
        lambda x: ' <sep> '.join(x.unique())
    ).reset_index()
    diagnoses_grouped = diagnoses_grouped.rename(columns={'icd_code_3digit': 'diagnoses'})
    
    # Merge all data
    df = admissions.merge(patients, on='subject_id', how='inner')
    df = df.merge(diagnoses_grouped, on=['subject_id', 'hadm_id'], how='left')
    df['diagnoses'] = df['diagnoses'].fillna('UNKNOWN')

    # Map shifted admission year to real calendar year (MIMIC-IV uses date shifting for de-identification)
    shifted_admit_year = df['admittime'].apply(lambda x: x.year)
    if 'real_anchor_year' in df.columns and df['real_anchor_year'].notna().any():
        real_year = df['real_anchor_year'] + (shifted_admit_year - df['anchor_year'])
        df['admit_year'] = real_year.round().fillna(shifted_admit_year).astype(int)
    else:
        df['admit_year'] = shifted_admit_year

    # Age at admission
    df['age'] = shifted_admit_year - df['anchor_year'] + df['anchor_age']
    
    # Filter: age between 18 and 89
    df = df[(df['age'] >= 18) & (df['age'] <= 89)]
    
    # Select relevant columns
    df = df[['subject_id', 'hadm_id', 'admit_year', 'age', 'gender', 'ethnicity', 'mortality', 'diagnoses']]
    
    # Save processed data
    processed_file = os.path.join(data_dir, 'mimic_iv_processed.csv')
    df.to_csv(processed_file, index=False)
    
    return processed_file


class MIMICIVStay:
    """Represents a single hospital admission."""
    
    def __init__(self, hadm_id, admit_year, mortality, age, gender, ethnicity, diagnoses):
        self.hadm_id = hadm_id
        self.admit_year = admit_year
        self.mortality = int(mortality)
        self.age = int(age)
        self.gender = gender
        self.ethnicity = ethnicity
        self.diagnoses = diagnoses  # String of space-separated ICD codes


def get_stay_dict(save_dir, force_reprocess=False):
    """Load processed data and create a dictionary of MIMICIVStay objects.
    
    Args:
        save_dir: Directory containing MIMIC-IV raw data and where to save outputs
        force_reprocess: If True, reprocess from raw CSVs even if cached files exist
    """
    mimic_dict = {}
    processed_file = os.path.join(save_dir, 'mimic_iv_processed.csv')
    if force_reprocess or not os.path.exists(processed_file):
        input_path = process_mimic_iv_data(save_dir)
    else:
        input_path = processed_file
    
    
    df = pd.read_csv(input_path)
    for _, row in df.iterrows():
        stay = MIMICIVStay(
            hadm_id=str(row['hadm_id']),
            admit_year=int(row['admit_year']),
            mortality=row['mortality'],
            age=row['age'],
            gender=row['gender'],
            ethnicity=row['ethnicity'],
            diagnoses=row['diagnoses']
        )
        mimic_dict[str(row['hadm_id'])] = stay
    
    # Save dictionary
    pickle.dump(mimic_dict, open(os.path.join(save_dir, 'mimic_iv_stay_dict.pkl'), 'wb'))
    return mimic_dict


def preprocess_mimic_iv(data_dir, force_reprocess=False):
    """
    Preprocess MIMIC-IV data into temporal splits by year.
    Organizes data into TIME_PERIODS (3-year windows: 2008-2010, 2011-2013, 2014-2016, 2017-2019, 2020-2022).
    
    Args:
        data_dir: Directory containing MIMIC-IV data
        force_reprocess: If True, rebuild stay dict and temporal splits from scratch
    """
    set_seed(0)
    np.random.seed(0)
    
    # Load stay dictionary
    stay_dict_path = os.path.join(data_dir, 'mimic_iv_stay_dict.pkl')
    if not os.path.exists(stay_dict_path) or force_reprocess:
        get_stay_dict(data_dir, force_reprocess=force_reprocess)
    
    data = pickle.load(open(stay_dict_path, 'rb'))
    
    # Organize data by time period
    datasets = {}
    for period in TIME_PERIODS:
        datasets[period] = {'train': {'diagnoses': [], 'labels': []}, 
                           'test': {'diagnoses': [], 'labels': []}}
    
    # Assign samples to time periods based on 3-year windows
    # Each period covers [p, p+3): 2008-2010, 2011-2013, 2014-2016, 2017-2019, 2020-2022
    for hadm_id, stay in data.items():
        period = None
        for p in TIME_PERIODS:
            if p <= stay.admit_year < p + 3:
                period = p
                break
        if period is None:
            continue
        
        # Split into train/test
        if np.random.rand() < (1 - TEST_SPLIT):
            split = 'train'
        else:
            split = 'test'
        
        datasets[period][split]['diagnoses'].append(stay.diagnoses)
        datasets[period][split]['labels'].append(stay.mortality)
    
    # Convert to numpy arrays
    for period in datasets:
        for split in datasets[period]:
            datasets[period][split]['diagnoses'] = np.array(datasets[period][split]['diagnoses'])
            datasets[period][split]['labels'] = np.array(datasets[period][split]['labels'])
    
    # Save preprocessed data
    output_file = os.path.join(data_dir, 'mimic_iv_wildtime.pkl')
    pickle.dump(datasets, open(output_file, 'wb'))
    
    return datasets


# Cache loaded pkl by path to avoid loading twice (saves memory when creating train+test)
_mimic_pkl_cache = {}

class MIMICIVDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-IV mortality prediction with temporal splits.
    Supports wild time learning across 5 time periods: 2008-2010, 2011-2013, 2014-2016, 2017-2019, 2020-2022.
    """
    
    def __init__(self, data_dir, split='train'):
        """
        Args:
            data_dir: Directory containing processed MIMIC-IV data
            split: 'train' or 'test'
        """
        self.data_dir = data_dir
        self.split = split
        self.mini_batch_size = 64
        
        # Preprocess if needed, or reprocess if cached periods don't match TIME_PERIODS
        data_file = os.path.join(data_dir, 'mimic_iv_wildtime.pkl')
        if not os.path.exists(data_file):
            preprocess_mimic_iv(data_dir)
        else:
            # Check if cached periods match current TIME_PERIODS; reprocess if not
            with open(data_file, 'rb') as f:
                cached = pickle.load(f)
            if sorted(cached.keys()) != sorted(TIME_PERIODS):
                preprocess_mimic_iv(data_dir, force_reprocess=True)
                data_file_key = os.path.abspath(data_file)
                _mimic_pkl_cache.pop(data_file_key, None)  # clear stale cache
        
        # Use cache to avoid loading pkl twice (train+test = 2x memory without this)
        data_file_key = os.path.abspath(data_file)
        if data_file_key not in _mimic_pkl_cache:
            with open(data_file, 'rb') as f:
                _mimic_pkl_cache[data_file_key] = pickle.load(f)
        self.datasets = _mimic_pkl_cache[data_file_key]
        self.time_periods = sorted(list(self.datasets.keys()))
        
        # Validate that all expected periods were loaded
        if len(self.time_periods) == 0:
            raise ValueError("No data found for any time period")
        
        # Initialize with first time period
        self.current_period = self.time_periods[0]
        self.num_classes = 2
        
        # Track which samples belong to which class for each period
        self.class_indices = {}
        for period in self.time_periods:
            self.class_indices[period] = {
                0: np.where(self.datasets[period][split]['labels'] == 0)[0],
                1: np.where(self.datasets[period][split]['labels'] == 1)[0]
            }
        
        print(f"Loaded MIMIC-IV data with {len(self.time_periods)} time periods: {self.time_periods}")
        for period in self.time_periods:
            n_samples = len(self.datasets[period][split]['labels'])
            n_positive = np.sum(self.datasets[period][split]['labels'])
            mortality_rate = (n_positive / n_samples * 100) if n_samples > 0 else 0
            print(f"Period {period}: {n_samples} samples ({n_positive} positive, {mortality_rate:.1f}% mortality)")
    
    def set_period(self, period):
        """Switch to a different time period."""
        if period not in self.time_periods:
            raise ValueError(f"Period {period} not available. Available periods: {self.time_periods}")
        self.current_period = period
    
    def __getitem__(self, idx):
        diagnoses = self.datasets[self.current_period][self.split]['diagnoses'][idx]
        label = int(self.datasets[self.current_period][self.split]['labels'][idx])
        
        return {
            'diagnoses': diagnoses,
            'label': torch.LongTensor([label])
        }
    
    def __len__(self):
        return len(self.datasets[self.current_period][self.split]['labels'])
    
    def get_num_periods(self):
        """Return number of time periods."""
        return len(self.time_periods)
    
    def get_periods(self):
        """Return list of available time periods."""
        return self.time_periods

    def get_period_stats(self):
        """Return summary statistics per time period (for domain validation / TDG analysis)."""
        stats = []
        for period in self.time_periods:
            labels = self.datasets[period][self.split]['labels']
            n = len(labels)
            n_positive = int(np.sum(labels))
            mortality_rate = (n_positive / n * 100) if n > 0 else 0
            stats.append({
                'period': period,
                'n_samples': n,
                'n_positive': n_positive,
                'mortality_rate': mortality_rate
            })
        return stats


def main():
    """CLI entry point for processing MIMIC-IV data."""
    import argparse
    parser = argparse.ArgumentParser(description='Process MIMIC-IV data for temporal mortality prediction')
    parser.add_argument('data_dir', help='Directory containing admissions.csv, patients.csv, diagnoses_icd.csv')
    parser.add_argument('--force', action='store_true', help='Force reprocessing even if cached files exist')
    args = parser.parse_args()
    preprocess_mimic_iv(args.data_dir, force_reprocess=args.force)
    print(f"Preprocessing complete. Output: {os.path.join(args.data_dir, 'mimic_iv_wildtime.pkl')}")


if __name__ == '__main__':
    main()
