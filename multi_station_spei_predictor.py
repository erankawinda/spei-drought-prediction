import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Ensure reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class ImprovedSPEIPredictor:
    """Enhanced SPEI predictor with time-series CV, tuned RF baseline, improved LSTM, and ensemble."""
    def __init__(self, seq_length=24):
        self.seq_length = seq_length
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.models = {}
        self.results = {}

    def load_data(self, path):
        print(f"Loading {path}...")
        df = pd.read_csv(path)
        # Clean columns
        df.columns = df.columns.str.strip().str.replace(' ', '')
        # Rename SPEI columns
        df.rename(columns={
            'spei_1':'spei1','spei_3':'spei3','spei_6':'spei6',
            'spei_9':'spei9','spei_12':'spei12'
        }, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        print(f"  Records: {len(df)}, Date: {df['date'].min().date()} to {df['date'].max().date()}\n")
        return df

    def create_features(self, df, target):
        df = df.copy()
        # Seasonal cycles
        df['month'] = df['date'].dt.month
        df['month_sin'] = np.sin(2*np.pi*df['month']/12)
        df['month_cos'] = np.cos(2*np.pi*df['month']/12)
        # Lag features
        for lag in [1,3,6,12]:
            df[f'{target}_lag_{lag}'] = df[target].shift(lag)
            df[f'mean_prep_lag_{lag}'] = df['mean_prep'].shift(lag)
            df[f'mean_tmp_lag_{lag}'] = df['mean_tmp'].shift(lag)
        # Rolling statistics
        for w in [3,6,12]:
            df[f'{target}_ma_{w}'] = df[target].rolling(w).mean()
            df[f'mean_prep_ma_{w}'] = df['mean_prep'].rolling(w).mean()
            df[f'mean_tmp_ma_{w}'] = df['mean_tmp'].rolling(w).mean()
        # Engineered features
        df['prep_evap_ratio'] = df['mean_prep']/(df['pot_evap']+1e-3)
        df['temp_range'] = df['max_tmp'] - df['min_tmp']
        df['cum_deficit'] = (df['mean_prep'] - df['pot_evap']).rolling(6).sum()
        df.dropna(inplace=True)
        return df.reset_index(drop=True)

    def build_lstm(self, n_features):
        model = keras.Sequential([
            layers.Input((self.seq_length, n_features)),
            layers.Conv1D(64, 3, activation='relu'),
            layers.LSTM(128, return_sequences=True),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.LSTM(64),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)
        ])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='mse', metrics=['mae']
        )
        return model

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        nse = 1 - np.sum((y_true-y_pred)**2)/np.sum((y_true-np.mean(y_true))**2)
        return {'RMSE':rmse,'MAE':mae,'R2':r2,'NSE':nse}

    def train(self, df, target):
        print(f"=== Training for {target} ===")
        df_feat = self.create_features(df, target)
        feat_cols = [c for c in df_feat.columns if c not in ['date','spei1','spei3','spei6','spei9','spei12']]
        X = df_feat[feat_cols].values
        y = df_feat[target].values

        # Scale
        Xs = self.scaler_X.fit_transform(X)
        ys = self.scaler_y.fit_transform(y.reshape(-1,1)).flatten()

        # Prepare LSTM sequences
        X_seq, y_seq = [], []
        for i in range(len(Xs)-self.seq_length):
            X_seq.append(Xs[i:i+self.seq_length])
            y_seq.append(ys[i+self.seq_length])
        X_seq = np.array(X_seq); y_seq = np.array(y_seq)
        split = int(0.8*len(X_seq))
        X_tr, X_te = X_seq[:split], X_seq[split:]
        y_tr, y_te = y_seq[:split], y_seq[split:]

        # --- Random Forest with TimeSeriesSplit ---
        tscv = TimeSeriesSplit(n_splits=5)
        rf = RandomForestRegressor(random_state=42)
        param_dist = {'n_estimators':[100,200,500], 'max_depth':[None,10,20],
                      'min_samples_split':[2,5,10]}
        rf_search = RandomizedSearchCV(rf, param_dist, n_iter=8, cv=tscv,
                                       scoring='neg_mean_squared_error', n_jobs=-1)
        # Use flat features offset by seq_length for RF
        X_flat = Xs[self.seq_length:]
        y_flat = y[self.seq_length:]
        rf_search.fit(X_flat, y_flat)
        best_rf = rf_search.best_estimator_
        rf_pred = best_rf.predict(X_flat)
        rf_metrics = self.evaluate(y_flat, rf_pred)
        self.models[f'rf_{target}'] = best_rf
        self.results[f'rf_{target}'] = {'metrics':rf_metrics,'y_true':y_flat,'y_pred':rf_pred}
        print(f"RF {target}: {rf_metrics}")

        # --- LSTM training ---
        lstm = self.build_lstm(n_features=len(feat_cols))
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=0.5,patience=5)
        ]
        history = lstm.fit(
            X_tr,y_tr,validation_data=(X_te,y_te),
            epochs=100, batch_size=32, callbacks=callbacks, verbose=1
        )
        y_pred_seq = lstm.predict(X_te).flatten()
        # Inverse scale
        y_true_orig = self.scaler_y.inverse_transform(y_te.reshape(-1,1)).flatten()
        y_pred_orig = self.scaler_y.inverse_transform(y_pred_seq.reshape(-1,1)).flatten()
        lstm_metrics = self.evaluate(y_true_orig, y_pred_orig)
        self.models[f'lstm_{target}'] = lstm
        self.results[f'lstm_{target}'] = {
            'metrics':lstm_metrics,'y_true':y_true_orig,
            'y_pred':y_pred_orig,'history':history.history
        }
        print(f"LSTM {target}: {lstm_metrics}\n")

        # --- Ensemble ---
        # Align lengths: use overlapping portion of flat RF and LSTM predictions
        min_len = min(len(y_flat), len(y_pred_orig))
        ens_true = y_true_orig[-min_len:]
        ens_pred = (rf_pred[-min_len:] + y_pred_orig[-min_len:]) / 2
        ens_metrics = self.evaluate(ens_true, ens_pred)
        self.results[f'ensemble_{target}'] = {
            'metrics':ens_metrics,'y_true':ens_true,'y_pred':ens_pred
        }
        print(f"Ensemble {target}: {ens_metrics}\n")

        return self.results

    def plot(self, key):
        if key not in self.results:
            print(f"No results for {key}")
            return
        r = self.results[key]
        # Scatter
        plt.figure(figsize=(6,6))
        plt.scatter(r['y_true'], r['y_pred'], alpha=0.5)
        plt.plot([min(r['y_true']),max(r['y_true'])],
                 [min(r['y_true']),max(r['y_true'])],'r--')
        plt.title(f"Actual vs Predicted ({key})")
        plt.xlabel('True'); plt.ylabel('Pred')
        plt.show()
        # Loss curve for LSTM
        if key.startswith('lstm'):
            hist = self.results[key]['history']
            plt.figure()
            plt.plot(hist['loss'],label='train_loss')
            plt.plot(hist['val_loss'],label='val_loss')
            plt.title(f"LSTM Training History ({key})")
            plt.legend(); plt.show()


def main():
    data_dir = 'data'
    files = {
        'Buttala':os.path.join(data_dir,'Buttala_speiall.csv'),
        'Padaviya':os.path.join(data_dir,'Padaviya_speiall.csv'),
        'Tissamaharama':os.path.join(data_dir,'Tissamaharama_speiall.csv')
    }
    for station,path in files.items():
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue
        print(f"---- Station: {station} ----")
        df = ImprovedSPEIPredictor().load_data(path)
        for tgt in ['spei3','spei6']:
            results = ImprovedSPEIPredictor().train(df, tgt)
            ImprovedSPEIPredictor().plot(f'ensemble_{tgt}')

if __name__=='__main__':
    main()