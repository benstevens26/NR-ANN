"""
feature_preprocessing.py - Data loading and preprocessing for LENRI classification
"""

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import re


def extract_label_cf4(filename):
    if re.search(r"00_C_", filename):
        return 0  # Carbon
    elif re.search(r"00_F_", filename):
        return 1  # Fluorine
    elif re.search(r"_C_", filename):
        return 0
    elif re.search(r"_F_", filename):
        return 1
    else:
        raise ValueError(f"Unexpected filename format: {filename}")


def extract_label_ar_cf4(filename):
    if re.search(r"00_C_", filename):
        return 0  # Carbon
    elif re.search(r"00_F_", filename):
        return 1  # Fluorine
    elif re.search(r"00_Ar_", filename):
        return 2  # Argon
    else:
        raise ValueError(f"Unexpected filename format: {filename}")


class NuclearRecoilDatasetCF4(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.drop(
            columns=["file_name", "label"]
        ).values  # Drop file_name explicitly
        self.labels = dataframe["label"].values

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # Class indices
        return x, y


class NuclearRecoilDatasetArCF4(Dataset):
    def __init__(self, dataframe):
        self.features = dataframe.drop(
            columns=["file_name", "label"]
        ).values  # Drop file_name explicitly
        self.labels = dataframe["label"].values  # Class indices

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)  # Class indices
        return x, y


def get_dataloaders_cf4(csv_file, batch_size=32, verbose=False):
    df = pd.read_csv(csv_file)
    df["label"] = df["file_name"].apply(extract_label_cf4)
    train, test = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=42
    )
    train, val = train_test_split(
        train, test_size=0.1765, stratify=train["label"], random_state=42
    )
    train_dataset = NuclearRecoilDatasetCF4(train)
    val_dataset = NuclearRecoilDatasetCF4(val)
    test_dataset = NuclearRecoilDatasetCF4(test)

    if verbose:
        return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        train_dataset,
        val_dataset, 
        test_dataset
        )   

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),

    )


def get_dataloaders_ar_cf4(csv_file, batch_size=32, verbose=False):
    df = pd.read_csv(csv_file)
    df["label"] = df["file_name"].apply(extract_label_ar_cf4)
    train, test = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=42
    )
    train, val = train_test_split(
        train, test_size=0.1765, stratify=train["label"], random_state=42
    )
    train_dataset = NuclearRecoilDatasetArCF4(train)
    val_dataset = NuclearRecoilDatasetArCF4(val)
    test_dataset = NuclearRecoilDatasetArCF4(test)

    if verbose:
        return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        train_dataset,
        val_dataset,
        test_dataset
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def get_dataloaders_cf4_biased(csv_file, batch_size=32, verbose=False):
    df = pd.read_csv(csv_file)
    df["label"] = df["file_name"].apply(extract_label_cf4)
    train, test = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=42
    )
    train, val = train_test_split(
        train, test_size=0.1765, stratify=train["label"], random_state=42
    )

    # Adjust the test set to have a 1:7.33 Carbon:Fluorine ratio
    carbon_test = test[test["label"] == 0]
    fluorine_test = test[test["label"] == 1]

    num_fluorine = len(fluorine_test)
    num_carbon = int(num_fluorine / 7.33)

    carbon_test = carbon_test.sample(n=num_carbon, random_state=42)
    test_biased = pd.concat([carbon_test, fluorine_test])

    # Print the number of Carbon and Fluorine in the test set
    if verbose:
        print(f"Number of Carbon in test set: {len(carbon_test)}")
        print(f"Number of Fluorine in test set: {len(fluorine_test)}")

    train_dataset = NuclearRecoilDatasetCF4(train)
    val_dataset = NuclearRecoilDatasetCF4(val)
    test_dataset = NuclearRecoilDatasetCF4(test_biased)

    if verbose:
        return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        train_dataset,
        val_dataset,
        test_dataset
    )
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )


def get_dataloaders_ar_cf4_biased(csv_file, batch_size=32, verbose=False):
    df = pd.read_csv(csv_file)
    df["label"] = df["file_name"].apply(extract_label_ar_cf4)
    train, test = train_test_split(
        df, test_size=0.15, stratify=df["label"], random_state=42
    )
    train, val = train_test_split(
        train, test_size=0.1765, stratify=train["label"], random_state=42
    )
    train_dataset = NuclearRecoilDatasetArCF4(train)
    val_dataset = NuclearRecoilDatasetArCF4(val)
    test_dataset = NuclearRecoilDatasetArCF4(test)

    # Adjust the test set to ratio C:Ar:F 1 : 0.94 : 7.33
    carbon_test = test[test["label"] == 0]
    fluorine_test = test[test["label"] == 1]
    argon_test = test[test["label"] == 2]

    num_fluorine = len(fluorine_test)
    num_carbon = int(num_fluorine * (1 / 7.33))
    num_argon = int(num_fluorine * (0.94 / 7.33))

    carbon_test = carbon_test.sample(n=num_carbon, random_state=42)
    argon_test = argon_test.sample(n=num_argon, random_state=42)
    test_biased = pd.concat([carbon_test, fluorine_test, argon_test])

    # Print the number of Carbon, Fluorine, and Argon in the test set
    if verbose:
        print(f"Number of Carbon in test set: {len(carbon_test)}")
        print(f"Number of Fluorine in test set: {len(fluorine_test)}")
        print(f"Number of Argon in test set: {len(argon_test)}")

    train_dataset = NuclearRecoilDatasetArCF4(train)
    val_dataset = NuclearRecoilDatasetArCF4(val)
    test_dataset = NuclearRecoilDatasetArCF4(test_biased)

    if verbose:
        return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        train_dataset,
        val_dataset,
        test_dataset
    )

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
    )