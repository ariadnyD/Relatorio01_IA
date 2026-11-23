import random
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fixa sementes para que os resultados sejam reproduzíveis."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_dataloaders(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    task: str,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Transforma os arrays em tensores do PyTorch e cria os DataLoaders."""
    set_seed(seed)

    x_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    x_val_t = torch.tensor(X_val, dtype=torch.float32)
    x_te_t = torch.tensor(X_te, dtype=torch.float32)

    if task in ("regression", "classification"):
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32).view(-1, 1)
        y_val_t = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
        y_te_t = torch.tensor(y_te, dtype=torch.float32).view(-1, 1)
    else:
        raise ValueError("task deve ser 'regression' ou 'classification'.")

    train_ds = TensorDataset(x_tr_t, y_tr_t)
    val_ds = TensorDataset(x_val_t, y_val_t)
    test_ds = TensorDataset(x_te_t, y_te_t)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def prepare_dataset(
    csv_path: str,
    task: str,
    target_col: str = "TARGET",
    test_size: float = 0.20,
    val_size_from_train: float = 0.20,
    random_state: int = 42,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """
    Faz a divisão treino/validação/teste, padroniza sem vazamento de dados
    e devolve os DataLoaders e scalers prontos.
    """
    set_seed(random_state)
    df = pd.read_csv(csv_path)

    X = df.drop(columns=[target_col]).to_numpy()
    y = df[target_col].to_numpy()

    stratify_all = y if task == "classification" else None

    # 1. 80% treino_completo / 20% teste (estratificado na classificação)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_all,
    )

    # 2. Dentro do treino_completo, separa 20% para validação
    stratify_train = y_train_full if task == "classification" else None
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=val_size_from_train,
        random_state=random_state,
        stratify=stratify_train,
    )

    # 3. Padroniza X usando APENAS X_train (evita vazamento de dados)
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_val_scaled = feature_scaler.transform(X_val)
    X_test_scaled = feature_scaler.transform(X_test)

    # 4. Na regressão, padroniza y usando APENAS y_train
    target_scaler = None
    if task == "regression":
        target_scaler = StandardScaler()
        y_train_proc = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_proc = target_scaler.transform(y_val.reshape(-1, 1)).ravel()
        y_test_proc = target_scaler.transform(y_test.reshape(-1, 1)).ravel()
    else:
        y_train_proc = y_train
        y_val_proc = y_val
        y_test_proc = y_test

    # 5. Cria os DataLoaders
    train_loader, val_loader, test_loader = make_dataloaders(
        X_train_scaled,
        y_train_proc,
        X_val_scaled,
        y_val_proc,
        X_test_scaled,
        y_test_proc,
        task=task,
        batch_size=batch_size,
        seed=random_state,
    )

    return {
        "feature_names": list(df.drop(columns=[target_col]).columns),
        "n_features": X.shape[1],
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
        "loaders": {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader,
        },
        "raw_splits": {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
        "scaled_splits": {
            "X_train": X_train_scaled,
            "X_val": X_val_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train_proc,
            "y_val": y_val_proc,
            "y_test": y_test_proc,
        },
    }


def inverse_target(y_scaled: np.ndarray, target_scaler: StandardScaler) -> np.ndarray:
    """Volta os valores previstos da regressão para a escala original."""
    arr = np.asarray(y_scaled).reshape(-1, 1)
    return target_scaler.inverse_transform(arr).ravel()