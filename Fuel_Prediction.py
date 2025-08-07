import pandas as pd
import numpy as np
import warnings
from scipy.stats import skew, kurtosis

# Sklearn Imports
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE

# Gradient Boosting Imports
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# PyTorch Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Suppress all warnings for clean output
warnings.filterwarnings('ignore')

# --- 3. Configuration & Setup ---
# Dynamically configure devices for GPU-accelerated libraries
IS_CUDA_AVAILABLE = torch.cuda.is_available()
DEVICE = torch.device("cuda" if IS_CUDA_AVAILABLE else "cpu")
LGBM_DEVICE = 'gpu' if IS_CUDA_AVAILABLE else 'cpu'
print(f"Execution device detected: {DEVICE}")
if IS_CUDA_AVAILABLE:
    print("GPU acceleration is enabled for PyTorch, CatBoost, LightGBM, and XGBoost.")

# K-Fold Cross-validation setup
N_SPLITS = 5
KF = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# --- 4. Data Loading ---
try:
    # Use the specified Kaggle file paths
    train_df = pd.read_csv('/kaggle/input/shell-dataset/train.csv')
    test_df = pd.read_csv('/kaggle/input/shell-dataset/test.csv')
except FileNotFoundError:
    print("FATAL ERROR: Specified Kaggle dataset files not found.")
    exit()

# --- 5. Advanced Feature Engineering ---
def create_all_features(df):
    """Creates a wide array of statistical and interaction features."""
    for i in range(1, 6):
        for j in range(1, 11):
            df[f'frac{i}_prop{j}'] = df[f'Component{i}_fraction'] * df[f'Component{i}_Property{j}']
    for j in range(1, 11):
        prop_cols = [f'Component{i}_Property{j}' for i in range(1, 6)]
        df[f'mean_prop{j}'] = df[prop_cols].mean(axis=1)
        df[f'std_prop{j}'] = df[prop_cols].std(axis=1)
        df[f'skew_prop{j}'] = df[prop_cols].skew(axis=1)
        df[f'kurt_prop{j}'] = df[prop_cols].kurtosis(axis=1)
    return df

print("Creating advanced features...")
train_df_eng = create_all_features(train_df.copy())
test_df_eng = create_all_features(test_df.copy())
print("Feature engineering complete.")

# --- 6. Feature Selection & Preprocessing ---
TARGETS = [f'BlendProperty{i}' for i in range(1, 11)]
test_ids = test_df['ID']
train_labels = train_df[TARGETS]

# Isolate original vs. engineered features
original_features = [col for col in train_df.columns if col not in ['ID'] + TARGETS]
engineered_feature_names = [col for col in train_df_eng.columns if col not in train_df.columns]

# Align columns and fill NaNs
X_orig = train_df_eng[original_features].copy().fillna(0)
X_test_orig = test_df_eng[original_features].copy().fillna(0)
X_eng = train_df_eng[engineered_feature_names].copy().fillna(0)
X_test_eng = test_df_eng[engineered_feature_names].copy().fillna(0)

# --- 7. Autoencoder (Embedding) and PCA Feature Generation ---
# Define Autoencoder to create embeddings
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, encoding_dim))
    def forward(self, x):
        return self.encoder(x)

# Define the new, more powerful Neural Network
class EvenMorePowerfulNN(nn.Module):
    def __init__(self, input_dim):
        super(EvenMorePowerfulNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.model(x)

# Scale original features (base for AE and PCA)
scaler_orig = StandardScaler()
X_scaled = scaler_orig.fit_transform(X_orig)
X_test_scaled = scaler_orig.transform(X_test_orig)

# Train Autoencoder to get X_embed
print("Training Autoencoder for feature embeddings...")
autoencoder = Autoencoder(X_scaled.shape[1]).to(DEVICE)
with torch.no_grad():
    X_embed = autoencoder(torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE)).cpu().numpy()
    X_test_embed = autoencoder(torch.tensor(X_test_scaled, dtype=torch.float32).to(DEVICE)).cpu().numpy()
print("Embeddings generated.")

# Generate PCA features with n_components=16 as requested
print("Generating PCA features...")
pca = PCA(n_components=16)
X_pca = pca.fit_transform(X_scaled)
X_test_pca = pca.transform(X_test_scaled)
print(f"PCA generated {X_pca.shape[1]} components.")

# --- 8. Assigned Model Training Loop ---
final_predictions = np.zeros((len(test_df), len(TARGETS)))
overall_best_mapes = []

# Define the model assignment map
model_assignment = {
    1: Ridge(alpha=1.0), 2: Ridge(alpha=1.0), 4: Ridge(alpha=1.0), 6: Ridge(alpha=1.0),
    5: RandomForestRegressor(n_estimators=500, max_depth=10, random_state=42, n_jobs=-1),
    3: "StackingEnsemble", 7: "StackingEnsemble", 8: "StackingEnsemble", 9: "StackingEnsemble",
    10: HuberRegressor()
}

for i, target_name in enumerate(TARGETS):
    prop_num = i + 1
    print(f"\n{'='*30}\nProcessing Target: {target_name} ({prop_num}/{len(TARGETS)})\n{'='*30}")
    y_target = train_labels[target_name].values

    # --- Per-Target Model-Based Feature Selection (RFE) ---
    print("Selecting best 20 engineered features for this target using RFE...")
    scaler_eng = StandardScaler()
    X_eng_scaled = scaler_eng.fit_transform(X_eng)
    X_test_eng_scaled = scaler_eng.transform(X_test_eng)
    
    # Use a fast model like LGBM for RFE
    rfe_estimator = LGBMRegressor(random_state=42, device=LGBM_DEVICE, verbosity=-1)
    selector = RFE(estimator=rfe_estimator, n_features_to_select=20, step=0.1, verbose=0)
    
    X_kbest_eng = selector.fit_transform(X_eng_scaled, y_target)
    X_test_kbest_eng = selector.transform(X_test_eng_scaled)
    print("Feature selection complete.")

    # Combine all features into the final feature set FOR THIS TARGET
    X_all = np.hstack([X_scaled, X_embed, X_pca, X_kbest_eng])
    X_test_all = np.hstack([X_test_scaled, X_test_embed, X_test_pca, X_test_kbest_eng])
    print(f"Final combined feature shape for this target: {X_all.shape}")

    # Get the assigned model for this target
    assigned_model = model_assignment[prop_num]
    model_name = assigned_model if isinstance(assigned_model, str) else assigned_model.__class__.__name__
    print(f"Assigned Model: {model_name}")

    oof_preds = np.zeros(X_all.shape[0])
    test_preds = np.zeros(X_test_all.shape[0])

    for fold, (train_idx, val_idx) in enumerate(KF.split(X_all, y_target)):
        X_train_fold, X_val_fold = X_all[train_idx], X_all[val_idx]
        y_train_fold, y_val_fold = y_target[train_idx], y_target[val_idx]

        if model_name == "StackingEnsemble":
            # --- Stacking Logic for difficult targets (3, 7, 8, 9) ---
            stack_base_models = {
                'LGBM': LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=20, random_state=42, device=LGBM_DEVICE, verbosity=-1),
                'CatBoost': CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=7, verbose=0, random_state=42, task_type='GPU' if IS_CUDA_AVAILABLE else 'CPU'),
                'RF': RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1),
                'NN': 'custom_nn'
            }
            meta_model = XGBRegressor(n_estimators=1000, learning_rate=0.01, max_depth=4, random_state=42, tree_method='hist', device=DEVICE, early_stopping_rounds=50)
            meta_features_val = np.zeros((len(X_val_fold), len(stack_base_models)))
            meta_features_test_fold = np.zeros((len(X_test_all), len(stack_base_models)))
            for model_idx, (name, model) in enumerate(stack_base_models.items()):
                if name == 'NN':
                    nn_model = EvenMorePowerfulNN(X_all.shape[1]).to(DEVICE)
                    optimizer = optim.Adam(nn_model.parameters(), lr=0.0005)
                    criterion = nn.MSELoss()
                    train_loader = DataLoader(TensorDataset(torch.tensor(X_train_fold, dtype=torch.float32), torch.tensor(y_train_fold, dtype=torch.float32).view(-1, 1)), batch_size=256, shuffle=True)
                    for epoch in range(200):
                        nn_model.train()
                        for inputs, labels in train_loader:
                            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                            optimizer.zero_grad()
                            outputs = nn_model(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                    nn_model.eval()
                    with torch.no_grad():
                        meta_features_val[:, model_idx] = nn_model(torch.tensor(X_val_fold, dtype=torch.float32).to(DEVICE)).cpu().numpy().flatten()
                        meta_features_test_fold[:, model_idx] = nn_model(torch.tensor(X_test_all, dtype=torch.float32).to(DEVICE)).cpu().numpy().flatten()
                else:
                    model.fit(X_train_fold, y_train_fold)
                    meta_features_val[:, model_idx] = model.predict(X_val_fold)
                    meta_features_test_fold[:, model_idx] = model.predict(X_test_all)
            meta_model.fit(meta_features_val, y_val_fold, eval_set=[(meta_features_val, y_val_fold)], verbose=False)
            oof_preds[val_idx] = meta_model.predict(meta_features_val)
            test_preds += meta_model.predict(meta_features_test_fold) / N_SPLITS
            
        else: # Logic for regularly assigned models
            model = assigned_model
            model.fit(X_train_fold, y_train_fold)
            oof_preds[val_idx] = model.predict(X_val_fold)
            test_preds += model.predict(X_test_all) / N_SPLITS

    # --- Report Performance and Assign Predictions ---
    mape = mean_absolute_percentage_error(y_target, oof_preds)
    print(f"\n>>> Final MAPE for {target_name}: {mape:.6f}")
    overall_best_mapes.append(mape)
    final_predictions[:, i] = test_preds

# --- 9. Final Report and Submission ---
print(f"\n\n{'='*30}\n--- OVERALL FINAL REPORT ---\n{'='*30}")
print(f"Overall Final MAPE (Average of each property's assigned model MAPE): {np.mean(overall_best_mapes):.6f}")

submission_df = pd.DataFrame(final_predictions, columns=TARGETS)
submission_df.insert(0, 'ID', test_ids)
submission_df.to_csv('solution.csv', index=False)
print("\nSubmission file 'solution.csv' has been successfully created.")