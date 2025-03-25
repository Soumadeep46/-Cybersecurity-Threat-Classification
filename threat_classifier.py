#!/usr/bin/env python3
"""
Improved Cybersecurity Threat Classification using Machine Learning
Dataset: ISCX IDS 2012
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import xgboost as xgb
import shap
import joblib
from datetime import datetime
import tarfile
import glob
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from boruta import BorutaPy
import xml.etree.ElementTree as ET
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import RobustScaler
from scipy import stats
import csv
import re
import io

# Suppress warnings
warnings.filterwarnings('ignore')

class CybersecurityThreatClassifier:
    """
    An improved class for cybersecurity threat classification using machine learning.
    """
    
    def __init__(self, dataset_path, output_dir='output'):
        """
        Initialize the classifier.
        
        Args:
            dataset_path (str): Path to the ISCX IDS 2012 dataset archive or directory.
            output_dir (str): Directory to save outputs.
        """
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.models = {}
        self.best_model = None
        self.execution_times = {}
        self.class_weights = None
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def extract_dataset(self, extract_dir='./dataset_temp'):
        """
        Extract the ISCX IDS 2012 dataset from tar.gz archive or find files in directory.
        
        Args:
            extract_dir (str): Directory to extract files to.
            
        Returns:
            list: Paths to dataset files.
        """
        start_time = time.time()
        
        if not os.path.exists(extract_dir):
            os.makedirs(extract_dir)
            
        print(f"Looking for dataset files...")
        
        # Check if dataset_path is a tar.gz file
        if os.path.isfile(self.dataset_path) and self.dataset_path.endswith(('.tar.gz', '.tgz')):
            print(f"Extracting dataset from {self.dataset_path}...")
            
            # Extract the tar.gz file
            with tarfile.open(self.dataset_path, 'r:gz') as tar:
                tar.extractall(path=extract_dir)
        elif os.path.isdir(self.dataset_path):
            print(f"Using dataset directory: {self.dataset_path}")
            extract_dir = self.dataset_path
        else:
            print(f"Dataset path {self.dataset_path} is not a tar.gz file or directory.")
            extract_dir = os.path.dirname(self.dataset_path)
        
        # Look for XML files in the extracted directory
        xml_files = glob.glob(f"{extract_dir}/**/*.xml", recursive=True)
        
        # Look for CSV files as an alternative
        csv_files = glob.glob(f"{extract_dir}/**/*.csv", recursive=True)
        
        # Look for text files as another alternative
        txt_files = glob.glob(f"{extract_dir}/**/*.txt", recursive=True)
        
        # Print found files
        print(f"Found {len(xml_files)} XML files.")
        if xml_files:
            for file in xml_files[:5]:  # Show first 5 files
                print(f"  - {file}")
            if len(xml_files) > 5:
                print(f"  - ... and {len(xml_files) - 5} more")
                
        print(f"Found {len(csv_files)} CSV files.")
        if csv_files:
            for file in csv_files[:5]:
                print(f"  - {file}")
            if len(csv_files) > 5:
                print(f"  - ... and {len(csv_files) - 5} more")
                
        print(f"Found {len(txt_files)} TXT files.")
        if txt_files:
            for file in txt_files[:5]:
                print(f"  - {file}")
            if len(txt_files) > 5:
                print(f"  - ... and {len(txt_files) - 5} more")
        
        # Check if we found any data files
        data_files = xml_files + csv_files + txt_files
        if not data_files:
            raise ValueError("No data files (XML, CSV, or TXT) found in the dataset.")
            
        self.execution_times['dataset_extraction'] = time.time() - start_time
        return {'xml': xml_files, 'csv': csv_files, 'txt': txt_files}
    
    def safe_parse_xml(self, xml_file):
        """
        Safely parse an XML file, handling potential errors.
        
        Args:
            xml_file (str): Path to the XML file.
            
        Returns:
            ElementTree or None: Parsed XML tree or None if parsing failed.
        """
        try:
            tree = ET.parse(xml_file)
            return tree
        except ET.ParseError as e:
            print(f"XML parsing error in {os.path.basename(xml_file)}: {e}")
            
            # Try to fix common XML issues and parse again
            try:
                with open(xml_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Fix common XML issues
                # Replace invalid characters
                content = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', content)
                
                # Ensure proper XML structure
                if not content.startswith('<?xml'):
                    content = '<?xml version="1.0" encoding="UTF-8"?>\n' + content
                
                # Parse from fixed content
                tree = ET.parse(io.StringIO(content))
                print(f"Successfully fixed and parsed {os.path.basename(xml_file)}")
                return tree
            except Exception as e2:
                print(f"Failed to fix XML in {os.path.basename(xml_file)}: {e2}")
                return None
    
    def parse_xml_to_dataframe(self, xml_files, max_files=None):
        """
        Parse XML files from ISCX IDS 2012 dataset into a pandas DataFrame.
        
        Args:
            xml_files (list): List of XML file paths.
            max_files (int, optional): Maximum number of files to process.
            
        Returns:
            pandas.DataFrame: Parsed dataset.
        """
        start_time = time.time()
        
        if max_files is not None and max_files < len(xml_files):
            xml_files = xml_files[:max_files]
            print(f"Processing {max_files} out of {len(xml_files)} XML files...")
        else:
            print(f"Processing all {len(xml_files)} XML files...")
        
        all_flows = []
        
        for xml_file in xml_files:
            try:
                print(f"Parsing {os.path.basename(xml_file)}...")
                
                # Safely parse XML
                tree = self.safe_parse_xml(xml_file)
                if tree is None:
                    print(f"Skipping {os.path.basename(xml_file)} due to parsing errors.")
                    continue
                
                root = tree.getroot()
                
                # Check if we have the expected structure
                flows = root.findall('.//Flow')
                if not flows:
                    print(f"No Flow elements found in {os.path.basename(xml_file)}. Checking alternative structure...")
                    # Try alternative structures
                    flows = root.findall('.//*')
                
                print(f"Found {len(flows)} flow elements in {os.path.basename(xml_file)}")
                
                for flow in flows:
                    flow_dict = {}
                    
                    # Try to extract all available attributes and elements
                    # This is a more flexible approach that doesn't assume a specific structure
                    for child in flow.iter():
                        if child.tag != flow.tag and child.text and child.text.strip():
                            # Convert numeric values
                            try:
                                value = float(child.text)
                            except (ValueError, TypeError):
                                value = child.text
                            
                            flow_dict[child.tag] = value
                    
                    # Skip if we couldn't extract any meaningful data
                    if len(flow_dict) < 3:
                        continue
                    
                    # Try to determine if this is an attack or normal flow
                    # Look for common attack indicators in various fields
                    is_attack = False
                    
                    # Check tag field if it exists
                    if 'Tag' in flow_dict:
                        is_attack = flow_dict['Tag'].lower() != 'normal'
                    
                    # Check appName field if it exists
                    elif 'appName' in flow_dict:
                        attack_keywords = ['attack', 'exploit', 'malware', 'botnet', 'ddos', 'intrusion']
                        is_attack = any(keyword in str(flow_dict['appName']).lower() for keyword in attack_keywords)
                    
                    # Check protocol anomalies
                    elif 'protocolName' in flow_dict and 'sourcePort' in flow_dict and 'destinationPort' in flow_dict:
                        # Suspicious port combinations
                        suspicious_ports = [22, 23, 445, 1433, 3306, 3389]
                        is_attack = (int(flow_dict.get('sourcePort', 0)) in suspicious_ports or 
                                    int(flow_dict.get('destinationPort', 0)) in suspicious_ports)
                    
                    # Add label
                    flow_dict['Label'] = 1 if is_attack else 0
                    
                    all_flows.append(flow_dict)
                
                print(f"Extracted {len(all_flows)} flows so far.")
                
                # Break early if we have enough data
                if len(all_flows) >= 10000:
                    print(f"Reached {len(all_flows)} flows, which is sufficient for analysis.")
                    break
                    
            except Exception as e:
                print(f"Error processing {xml_file}: {e}")
                continue
        
        if not all_flows:
            print("No valid flows found in XML files. Trying to load CSV files instead...")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_flows)
        
        # Ensure we have numeric features
        for col in df.columns:
            if col != 'Label' and df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass  # Keep as object if conversion fails
        
        # Drop columns with too many missing values
        missing_threshold = 0.5
        cols_to_drop = [col for col in df.columns if df[col].isnull().sum() > missing_threshold * len(df)]
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} columns with >{missing_threshold*100}% missing values")
            df = df.drop(columns=cols_to_drop)
        
        # Add derived features if possible
        self.add_derived_features(df)
        
        print(f"Final dataset: {len(df)} flows with {df.shape[1]} features.")
        print("\nClass Distribution:")
        print(df['Label'].value_counts())
        
        self.data = df
        self.execution_times['xml_parsing'] = time.time() - start_time
        return df
    
    def parse_csv_to_dataframe(self, csv_files, max_files=None):
        """
        Parse CSV files into a pandas DataFrame.
        
        Args:
            csv_files (list): List of CSV file paths.
            max_files (int, optional): Maximum number of files to process.
            
        Returns:
            pandas.DataFrame: Parsed dataset.
        """
        start_time = time.time()
        
        if max_files is not None and max_files < len(csv_files):
            csv_files = csv_files[:max_files]
            print(f"Processing {max_files} out of {len(csv_files)} CSV files...")
        else:
            print(f"Processing all {len(csv_files)} CSV files...")
        
        all_data = []
        
        for csv_file in csv_files:
            try:
                print(f"Parsing {os.path.basename(csv_file)}...")
                
                # Try different delimiters and encodings
                for delimiter in [',', ';', '\t', '|']:
                    try:
                        # Try to read with different encodings
                        for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
                            try:
                                df = pd.read_csv(csv_file, delimiter=delimiter, encoding=encoding, error_bad_lines=False)
                                
                                # Check if we got a valid DataFrame with multiple columns
                                if df.shape[1] > 1:
                                    print(f"Successfully parsed {os.path.basename(csv_file)} with delimiter '{delimiter}' and encoding '{encoding}'")
                                    
                                    # Look for a label column
                                    label_candidates = ['label', 'class', 'attack', 'tag', 'type']
                                    label_col = None
                                    
                                    for col in df.columns:
                                        if col.lower() in label_candidates:
                                            label_col = col
                                            break
                                    
                                    # If no label column found, try to infer from values
                                    if label_col is None:
                                        for col in df.columns:
                                            unique_vals = df[col].astype(str).str.lower().unique()
                                            if len(unique_vals) <= 10 and any(val in ['normal', 'attack', 'malicious', 'benign'] for val in unique_vals):
                                                label_col = col
                                                break
                                    
                                    # If still no label column, use the last column
                                    if label_col is None:
                                        label_col = df.columns[-1]
                                        print(f"No clear label column found. Using {label_col} as the label.")
                                    
                                    # Convert label to binary (0 for normal, 1 for attack)
                                    df['Label'] = df[label_col].apply(
                                        lambda x: 0 if str(x).lower() in ['normal', 'benign', '0'] else 1
                                    )
                                    
                                    # Drop the original label column if it's different from 'Label'
                                    if label_col != 'Label':
                                        df = df.drop(columns=[label_col])
                                    
                                    all_data.append(df)
                                    break
                            except Exception as e:
                                continue
                        
                        # Break out of delimiter loop if we successfully parsed the file
                        if len(all_data) > 0 and all_data[-1].shape[0] > 0:
                            break
                    except:
                        continue
            except Exception as e:
                print(f"Error processing {csv_file}: {e}")
                continue
        
        if not all_data:
            print("No valid data found in CSV files.")
            return None
        
        # Combine all DataFrames
        try:
            # Try to concatenate if columns match
            df = pd.concat(all_data, ignore_index=True)
        except:
            # If columns don't match, use the largest DataFrame
            df = max(all_data, key=lambda x: x.shape[0])
            print(f"Using the largest DataFrame with {df.shape[0]} rows and {df.shape[1]} columns.")
        
        # Ensure we have numeric features
        for col in df.columns:
            if col != 'Label' and df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    pass
        
        # Drop columns with too many missing values
        missing_threshold = 0.5
        cols_to_drop = [col for col in df.columns if df[col].isnull().sum() > missing_threshold * len(df)]
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} columns with >{missing_threshold*100}% missing values")
            df = df.drop(columns=cols_to_drop)
        
        # Add derived features
        self.add_derived_features(df)
        
        print(f"Final dataset: {len(df)} rows with {df.shape[1]} features.")
        print("\nClass Distribution:")
        print(df['Label'].value_counts())
        
        self.data = df
        self.execution_times['csv_parsing'] = time.time() - start_time
        return df
    
    def add_derived_features(self, df):
        """
        Add derived features to the DataFrame.
        
        Args:
            df (pandas.DataFrame): DataFrame to add features to.
        """
        # Find columns related to bytes
        byte_cols = [col for col in df.columns if 'byte' in col.lower() or 'payload' in col.lower()]
        
        # Find columns related to packets
        packet_cols = [col for col in df.columns if 'packet' in col.lower()]
        
        # Find columns related to duration
        duration_cols = [col for col in df.columns if 'duration' in col.lower() or 'time' in col.lower()]
        
        # Add derived features if possible
        if len(byte_cols) >= 2:
            # Bytes ratio
            df['bytesRatio'] = df[byte_cols[0]] / (df[byte_cols[1]] + 1)
        
        if len(packet_cols) >= 2:
            # Packets ratio
            df['packetsRatio'] = df[packet_cols[0]] / (df[packet_cols[1]] + 1)
        
        if len(byte_cols) >= 1 and len(packet_cols) >= 1:
            # Average packet size
            df['avgPacketSize'] = df[byte_cols[0]] / (df[packet_cols[0]] + 1)
        
        if len(duration_cols) >= 1 and len(byte_cols) >= 1:
            # Bytes per second
            df['bytesPerSecond'] = df[byte_cols[0]] / (df[duration_cols[0]] / 1000000 + 0.001)
        
        if len(duration_cols) >= 1 and len(packet_cols) >= 1:
            # Packets per second
            df['packetsPerSecond'] = df[packet_cols[0]] / (df[duration_cols[0]] / 1000000 + 0.001)
    
    def create_synthetic_data(self, n_samples=10000):
        """
        Create synthetic data for demonstration when no real data is available.
        
        Args:
            n_samples (int): Number of samples to generate.
            
        Returns:
            pandas.DataFrame: Synthetic dataset.
        """
        print(f"Creating synthetic dataset with {n_samples} samples...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Generate synthetic features
        n_features = 20
        X = np.random.randn(n_samples, n_features)
        
        # Generate synthetic labels (binary classification: normal vs attack)
        # Create imbalanced dataset (80% normal, 20% attack)
        y = np.zeros(n_samples)
        attack_indices = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
        y[attack_indices] = 1
        
        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        
        # Create DataFrame
        df = pd.DataFrame(X, columns=feature_names)
        df['Label'] = y
        
        # Add some categorical features
        protocols = ['TCP', 'UDP', 'ICMP']
        df['Protocol_Type'] = np.random.choice(protocols, size=n_samples)
        
        services = ['http', 'ftp', 'smtp', 'ssh', 'dns']
        df['Service'] = np.random.choice(services, size=n_samples)
        
        # Add some network-specific features
        df['sourcePort'] = np.random.randint(1024, 65535, size=n_samples)
        df['destinationPort'] = np.random.randint(1, 1024, size=n_samples)
        df['flowDuration'] = np.random.exponential(scale=1000000, size=n_samples)
        df['totalSourceBytes'] = np.random.exponential(scale=10000, size=n_samples)
        df['totalDestinationBytes'] = np.random.exponential(scale=8000, size=n_samples)
        df['totalSourcePackets'] = np.random.exponential(scale=50, size=n_samples)
        df['totalDestinationPackets'] = np.random.exponential(scale=40, size=n_samples)
        
        # Add derived features
        df['bytesRatio'] = df['totalSourceBytes'] / (df['totalDestinationBytes'] + 1)
        df['packetsRatio'] = df['totalSourcePackets'] / (df['totalDestinationPackets'] + 1)
        df['sourceAvgPacketSize'] = df['totalSourceBytes'] / (df['totalSourcePackets'] + 1)
        df['destAvgPacketSize'] = df['totalDestinationBytes'] / (df['totalDestinationPackets'] + 1)
        df['bytesPerSecond'] = df['totalSourceBytes'] / (df['flowDuration'] / 1000000 + 0.001)
        df['packetsPerSecond'] = df['totalSourcePackets'] / (df['flowDuration'] / 1000000 + 0.001)
        
        print(f"Created synthetic dataset with {n_samples} samples and {df.shape[1]} features.")
        print("\nClass Distribution:")
        print(df['Label'].value_counts())
        
        self.data = df
        return df
    
    def preprocess_data(self, test_size=0.2, random_state=42):
        """
        Preprocess the data for model training.
        
        Args:
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        
        Returns:
            tuple: Preprocessed training and testing data.
        """
        start_time = time.time()
        
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        print("\nPreprocessing data...")
        
        # Drop rows with too many missing values
        missing_threshold = 0.5
        rows_before = len(self.data)
        self.data = self.data.dropna(thresh=int(self.data.shape[1] * (1 - missing_threshold)))
        rows_after = len(self.data)
        print(f"Dropped {rows_before - rows_after} rows with more than {missing_threshold*100}% missing values.")
        
        # Handle remaining missing values
        numerical_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        for col in numerical_cols:
            if self.data[col].isnull().sum() > 0:
                # Use median for skewed distributions
                if abs(stats.skew(self.data[col].dropna())) > 1:
                    self.data[col] = self.data[col].fillna(self.data[col].median())
                else:
                    self.data[col] = self.data[col].fillna(self.data[col].mean())
        
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.data[col].isnull().sum() > 0:
                self.data[col] = self.data[col].fillna(self.data[col].mode()[0])
        
        # Encode categorical features
        for col in categorical_cols:
            le = LabelEncoder()
            self.data[col] = le.fit_transform(self.data[col])
        
        # Separate features and target
        X = self.data.drop('Label', axis=1)
        y = self.data['Label']
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        # Remove outliers from training data
        print("Removing outliers from training data...")
        X_train_clean = X_train.copy()
        y_train_clean = y_train.copy()
        
        for col in X_train.select_dtypes(include=['float64', 'int64']).columns:
            q1 = X_train[col].quantile(0.01)
            q3 = X_train[col].quantile(0.99)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outlier_mask = (X_train[col] >= lower_bound) & (X_train[col] <= upper_bound)
            X_train_clean = X_train_clean[outlier_mask]
            y_train_clean = y_train_clean[outlier_mask]
        
        print(f"Removed {len(X_train) - len(X_train_clean)} outliers from training data.")
        
        # Scale features using RobustScaler (more resistant to outliers)
        scaler = RobustScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train_clean),
            columns=X_train_clean.columns
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )
        
        # Calculate class weights for imbalanced data
        unique_classes = np.unique(y_train_clean)
        self.class_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=y_train_clean
        )
        self.class_weights = dict(zip(unique_classes, self.class_weights))
        print(f"Class weights: {self.class_weights}")
        
        self.X_train = X_train_scaled
        self.y_train = y_train_clean
        self.X_test = X_test_scaled
        self.y_test = y_test
        self.feature_names = X.columns.tolist()
        
        self.execution_times['preprocessing'] = time.time() - start_time
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def handle_class_imbalance(self, method='hybrid', random_state=42):
        """
        Handle class imbalance in the training data.
        
        Args:
            method (str): Method to use ('smote', 'weighted', 'hybrid').
            random_state (int): Random seed for reproducibility.
            
        Returns:
            tuple: Resampled training data.
        """
        start_time = time.time()
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        print(f"\nHandling class imbalance using {method}...")
        print(f"Original class distribution: {pd.Series(self.y_train).value_counts()}")
        
        if method == 'smote':
            # Apply SMOTE to oversample the minority class
            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)
            
        elif method == 'weighted':
            # Use class weights (already computed in preprocess_data)
            # No resampling needed
            X_resampled, y_resampled = self.X_train, self.y_train
            
        elif method == 'hybrid':
            # Combine undersampling of majority class and oversampling of minority class
            # First undersample the majority class
            rus = RandomUnderSampler(sampling_strategy=0.5, random_state=random_state)
            X_under, y_under = rus.fit_resample(self.X_train, self.y_train)
            
            # Then apply SMOTE to balance the classes
            smote = SMOTE(random_state=random_state)
            X_resampled, y_resampled = smote.fit_resample(X_under, y_under)
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'smote', 'weighted', or 'hybrid'.")
        
        print(f"Resampled class distribution: {pd.Series(y_resampled).value_counts()}")
        
        self.X_train = X_resampled
        self.y_train = y_resampled
        
        self.execution_times['class_imbalance'] = time.time() - start_time
        return self.X_train, self.y_train
    
    def select_features_boruta(self, n_estimators=100, max_iter=100, random_state=42):
        """
        Select features using Boruta algorithm.
        
        Args:
            n_estimators (int): Number of estimators for the random forest.
            max_iter (int): Maximum number of iterations.
            random_state (int): Random seed for reproducibility.
            
        Returns:
            tuple: Reduced feature sets for training and testing.
        """
        start_time = time.time()
        
        if self.X_train is None or self.X_test is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        print("\nSelecting features using Boruta algorithm...")
        
        # Initialize Random Forest for Boruta
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
        
        # Initialize Boruta
        boruta = BorutaPy(
            rf,
            n_estimators=n_estimators,
            max_iter=max_iter,
            verbose=2,
            random_state=random_state
        )
        
        # Fit Boruta
        boruta.fit(self.X_train.values, self.y_train.values)
        
        # Get selected feature indices
        selected_indices = np.where(boruta.support_)[0]
        
        # Get tentative feature indices
        tentative_indices = np.where(boruta.support_weak_)[0]
        
        # Combine confirmed and tentative features
        all_selected_indices = np.union1d(selected_indices, tentative_indices)
        
        # Get selected feature names
        selected_features = [self.feature_names[i] for i in all_selected_indices]
        
        print(f"Selected {len(selected_features)} features:")
        for i, feature in enumerate(selected_features):
            print(f"  {i+1}. {feature}")
        
        # Update feature names and data
        self.feature_names = selected_features
        self.X_train = self.X_train.iloc[:, all_selected_indices]
        self.X_test = self.X_test.iloc[:, all_selected_indices]
        
        self.execution_times['feature_selection'] = time.time() - start_time
        return self.X_train, self.X_test
    
    def train_models(self, use_class_weights=True):
        """
        Train multiple models and select the best one.
        
        Args:
            use_class_weights (bool): Whether to use class weights for imbalanced data.
            
        Returns:
            dict: Trained models.
        """
        start_time = time.time()
        
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")
        
        print("\nTraining models...")
        
        # Define base models
        base_models = {
            'logistic_regression': {
                'model': LogisticRegression(max_iter=5000, random_state=42),  # Increased max_iter to fix convergence
                'params': {
                    'C': [0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': ['balanced', None] if use_class_weights else [None]
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', 'balanced_subsample', None] if use_class_weights else [None]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7, 9],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'scale_pos_weight': [1, 5, 10] if use_class_weights else [1]
                }
            }
        }
        
        # Train and evaluate each base model
        best_f1 = 0
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, config in base_models.items():
            print(f"\nTraining {name}...")
            
            # Hyperparameter tuning
            print(f"Performing hyperparameter tuning for {name}...")
            search = RandomizedSearchCV(
                config['model'],
                config['params'],
                n_iter=20,
                cv=cv,
                scoring='f1',  # Focus on F1 score for imbalanced data
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            search.fit(self.X_train, self.y_train)
            
            # Get best model
            best_model = search.best_estimator_
            self.models[name] = best_model
            
            # Evaluate on test set
            y_pred = best_model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='binary'
            )
            
            print(f"Best parameters for {name}: {search.best_params_}")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            
            # Save detailed classification report
            report = classification_report(self.y_test, y_pred)
            print(f"\nClassification Report for {name}:\n{report}")
            
            # Update best model if this one is better
            if f1 > best_f1:
                best_f1 = f1
                self.best_model = best_model
                self.best_model_name = name
        
        # Train stacking classifier
        print("\nTraining stacking classifier...")
        
        # Define base estimators for stacking
        estimators = [
            ('rf', self.models['random_forest']),
            ('xgb', self.models['xgboost'])
        ]
        
        # Define final estimator
        final_estimator = LogisticRegression(max_iter=5000, random_state=42)  # Increased max_iter
        
        # Create and train stacking classifier
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            stack_method='predict_proba'
        )
        
        stacking_clf.fit(self.X_train, self.y_train)
        self.models['stacking'] = stacking_clf
        
        # Evaluate stacking classifier
        y_pred = stacking_clf.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.y_test, y_pred, average='binary'
        )
        
        print(f"Stacking Classifier Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Save detailed classification report
        report = classification_report(self.y_test, y_pred)
        print(f"\nClassification Report for stacking classifier:\n{report}")
        
        # Update best model if stacking is better
        if f1 > best_f1:
            best_f1 = f1
            self.best_model = stacking_clf
            self.best_model_name = 'stacking'
        
        print(f"\nBest model: {self.best_model_name} with F1 Score: {best_f1:.4f}")
        
        self.execution_times['model_training'] = time.time() - start_time
        return self.models
    
    def evaluate_models(self):
        """
        Evaluate trained models and visualize results.
        
        Returns:
            dict: Evaluation metrics for each model.
        """
        start_time = time.time()
        
        if not self.models:
            raise ValueError("Models not trained. Call train_models() first.")
        
        print("\nEvaluating models...")
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                self.y_test, y_pred, average='binary'
            )
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(self.y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.savefig(f"{self.output_dir}/confusion_matrix_{name}.png")
            plt.close()
            
            # Feature importance (if applicable)
            if hasattr(model, 'feature_importances_'):
                plt.figure(figsize=(12, 8))
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.barh(range(len(importances)), importances[indices])
                plt.yticks(range(len(importances)), [self.feature_names[i] for i in indices])
                plt.title(f'Feature Importance - {name}')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/feature_importance_{name}.png")
                plt.close()
        
        # Compare models
        plt.figure(figsize=(12, 8))
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        x = np.arange(len(metrics))
        width = 0.2
        
        for i, (name, metrics_dict) in enumerate(results.items()):
            values = [metrics_dict[metric] for metric in metrics]
            plt.bar(x + i * width, values, width, label=name)
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Comparison')
        plt.xticks(x + width, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_comparison.png")
        plt.close()
        
        # SHAP values for explainability (for the best model)
        if self.best_model_name in ['random_forest', 'xgboost', 'stacking']:
            print(f"\nGenerating SHAP values for {self.best_model_name}...")
            
            try:
                # For stacking, use the first base estimator (random forest)
                if self.best_model_name == 'stacking':
                    model_for_shap = self.models['random_forest']
                else:
                    model_for_shap = self.best_model
                
                # Create explainer
                explainer = shap.TreeExplainer(model_for_shap)
                
                # Calculate SHAP values for a subset of test data
                shap_subset = min(100, len(self.X_test))
                X_test_sample = self.X_test.iloc[:shap_subset]
                
                # Get SHAP values
                shap_values = explainer.shap_values(X_test_sample)
                
                # Summary plot
                plt.figure(figsize=(12, 10))
                if isinstance(shap_values, list):
                    # For multi-class output
                    shap.summary_plot(
                        shap_values[1] if len(shap_values) > 1 else shap_values[0],
                        X_test_sample,
                        feature_names=self.feature_names,
                        show=False
                    )
                else:
                    # For single class output
                    shap.summary_plot(
                        shap_values,
                        X_test_sample,
                        feature_names=self.feature_names,
                        show=False
                    )
                
                plt.title(f'SHAP Summary Plot - {self.best_model_name}')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/shap_summary_{self.best_model_name}.png")
                plt.close()
                
                # Force plot for a single prediction - using updated SHAP API
                plt.figure(figsize=(20, 3))
                
                # Get expected value
                expected_value = explainer.expected_value
                if isinstance(expected_value, list):
                    expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
                
                # Get SHAP values for a single instance
                if isinstance(shap_values, list):
                    instance_shap_values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
                else:
                    instance_shap_values = shap_values[0]
                
                # Create force plot using updated API
                shap.plots.force(
                    expected_value,
                    instance_shap_values,
                    X_test_sample.iloc[0],
                    feature_names=self.feature_names,
                    matplotlib=True,
                    show=False
                )
                
                plt.title(f'SHAP Force Plot - {self.best_model_name}')
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/shap_force_{self.best_model_name}.png")
                plt.close()
                
            except Exception as e:
                print(f"Error generating SHAP values: {e}")
                # Fallback to feature importance plot if SHAP fails
                if hasattr(self.best_model, 'feature_importances_'):
                    plt.figure(figsize=(12, 8))
                    importances = self.best_model.feature_importances_
                    indices = np.argsort(importances)[::-1]
                    
                    plt.barh(range(len(importances)), importances[indices])
                    plt.yticks(range(len(importances)), [self.feature_names[i] for i in indices])
                    plt.title(f'Feature Importance (SHAP alternative) - {self.best_model_name}')
                    plt.tight_layout()
                    plt.savefig(f"{self.output_dir}/feature_importance_alt_{self.best_model_name}.png")
                    plt.close()
        
        self.execution_times['evaluation'] = time.time() - start_time
        return results
    
    def save_model(self):
        """
        Save the best model to disk.
        
        Returns:
            str: Path to the saved model.
        """
        if self.best_model is None:
            raise ValueError("No best model found. Call train_models() first.")
        
        model_path = f"{self.output_dir}/{self.best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(self.best_model, model_path)
        print(f"\nBest model ({self.best_model_name}) saved to {model_path}")
        
        # Save feature names for future use
        feature_names_path = f"{self.output_dir}/feature_names_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
        joblib.dump(self.feature_names, feature_names_path)
        print(f"Feature names saved to {feature_names_path}")
        
        return model_path
    
    def generate_report(self):
        """
        Generate a summary report of the analysis.
        
        Returns:
            str: Path to the report file.
        """
        report_path = f"{self.output_dir}/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        with open(report_path, 'w') as f:
            f.write("# Cybersecurity Threat Classification Report\n\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Dataset Information\n\n")
            f.write(f"Number of samples: {self.data.shape[0]}\n")
            f.write(f"Number of features: {len(self.feature_names)}\n")
            f.write(f"Class distribution:\n{self.data['Label'].value_counts().to_string()}\n\n")
            
            f.write("## Feature Selection\n\n")
            f.write(f"Selected features ({len(self.feature_names)}):\n")
            for i, feature in enumerate(self.feature_names):
                f.write(f"{i+1}. {feature}\n")
            f.write("\n")
            
            f.write("## Model Performance\n\n")
            for name, model in self.models.items():
                y_pred = model.predict(self.X_test)
                accuracy = accuracy_score(self.y_test, y_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    self.y_test, y_pred, average='binary'
                )
                
                f.write(f"### {name}\n\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"Precision: {precision:.4f}\n")
                f.write(f"Recall: {recall:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n\n")
                f.write(f"Classification Report:\n{classification_report(self.y_test, y_pred)}\n\n")
            
            f.write("## Best Model\n\n")
            f.write(f"Best model: {self.best_model_name}\n\n")
            
            f.write("## Execution Times\n\n")
            for step, duration in self.execution_times.items():
                f.write(f"{step}: {duration:.2f} seconds\n")
            f.write(f"Total: {sum(self.execution_times.values()):.2f} seconds\n")
        
        print(f"\nReport generated at {report_path}")
        return report_path
    
    def run_pipeline(self, max_files=None, imbalance_method='hybrid'):
        """
        Run the complete analysis pipeline.
        
        Args:
            max_files (int, optional): Maximum number of XML files to process.
            imbalance_method (str): Method to handle class imbalance ('smote', 'weighted', or 'hybrid').
            
        Returns:
            str: Path to the generated report.
        """
        print("Starting improved cybersecurity threat classification pipeline...")
        
        # Extract dataset
        data_files = self.extract_dataset()
        
        # Try to parse XML files
        if data_files['xml']:
            self.data = self.parse_xml_to_dataframe(data_files['xml'], max_files=max_files)
        
        # If XML parsing failed, try CSV files
        if self.data is None and data_files['csv']:
            self.data = self.parse_csv_to_dataframe(data_files['csv'], max_files=max_files)
        
        # If both XML and CSV parsing failed, create synthetic data
        if self.data is None:
            print("Could not parse any data files. Creating synthetic data for demonstration.")
            self.create_synthetic_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Handle class imbalance
        self.handle_class_imbalance(method=imbalance_method)
        
        # Select features using Boruta
        try:
            self.select_features_boruta()
        except Exception as e:
            print(f"Error in feature selection: {e}")
            print("Continuing with all features...")
        
        # Train models
        self.train_models()
        
        # Evaluate models
        self.evaluate_models()
        
        # Save best model
        self.save_model()
        
        # Generate report
        report_path = self.generate_report()
        
        print("\nPipeline completed successfully!")
        print(f"Total execution time: {sum(self.execution_times.values()):.2f} seconds")
        
        return report_path


if __name__ == "__main__":
    # Example usage
    classifier = CybersecurityThreatClassifier(
        dataset_path="./dataset_temp/iscxids2012-master",  # Use the extracted directory
        output_dir='output'
    )
    
    # Run the complete pipeline
    # Limit the number of files for faster processing if needed
    report_path = classifier.run_pipeline(max_files=5, imbalance_method='hybrid')
    
    print(f"\nAnalysis completed. Report available at: {report_path}")