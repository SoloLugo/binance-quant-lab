import os
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE_DIR = "/app"
FEATURES_DIR = os.path.join(BASE_DIR, "data", "features")
LABELS_DIR = os.path.join(BASE_DIR, "data", "labels")
CONFIG_FILE = os.path.join(BASE_DIR, "config", "base.yaml")

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def run_feature_importance(symbol):
    print(f"\n{'='*70}\n🔬 ABRIENDO LA CAJA NEGRA: {symbol}\n{'='*70}")
    
    features_path = os.path.join(FEATURES_DIR, f"symbol={symbol}", "features.parquet")
    labels_path = os.path.join(LABELS_DIR, f"symbol={symbol}", "labels.parquet")
    
    if not (os.path.exists(features_path) and os.path.exists(labels_path)):
        print(f"[!] Faltan datos para {symbol}.")
        return

    # 1. Cargar y ensamblar
    df_features = pd.read_parquet(features_path)
    df_labels = pd.read_parquet(labels_path)
    
    df_features.index = df_features.index.tz_localize(None)
    df_labels.index = df_labels.index.tz_localize(None)
    
    df = df_features.join(df_labels, how='inner')
    df = df[df['label'] != -2].dropna()
    df['target'] = (df['label'] == 1).astype(int)
    
    # 2. Filtrar las columnas que realmente ve el modelo (excluyendo contexto y etiquetas)
    context_cols = ['regime_bull', 'regime_bear', 'regime_high_vol']
    features_cols = [c for c in df_features.columns if c not in context_cols and c != 'label']
    
    X = df[features_cols]
    y = df['target']
    
    # 3. Estandarización obligatoria para comparar manzanas con manzanas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 4. Ajustar el modelo a toda la historia para extraer los pesos globales
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    
    # 5. Extraer e interpretar coeficientes
    coeficientes = model.coef_[0]
    
    importancia_df = pd.DataFrame({
        'Feature': features_cols,
        'Peso': coeficientes,
        'Importancia_Absoluta': np.abs(coeficientes)
    })
    
    # Ordenar desde la variable más dominante hasta el ruido inútil
    importancia_df = importancia_df.sort_values(by='Importancia_Absoluta', ascending=False).reset_index(drop=True)
    
    print(f"{'Ranking':<8} | {'Variable':<30} | {'Impacto':<10} | {'Dirección'}")
    print("-" * 70)
    
    for idx, row in importancia_df.iterrows():
        peso = row['Peso']
        feature = row['Feature']
        
        # Clasificamos visualmente el impacto
        if peso > 0:
            direccion = "🟢 Favorece LONG (Sube Probabilidad)"
        else:
            direccion = "🔴 Frena LONG / Indica RIESGO"
            
        print(f"#{idx+1:<7} | {feature:<30} | {peso:>+10.4f} | {direccion}")

    # Análisis de Ablación Rápido
    print(f"\n{'-'*70}")
    print("💡 DIAGNÓSTICO DEL INGENIERO:")
    top_3 = importancia_df['Feature'].head(3).tolist()
    useless = importancia_df[importancia_df['Importancia_Absoluta'] < 0.01]['Feature'].tolist()
    
    print(f"-> Los 3 pilares de tu Edge: {', '.join(top_3)}")
    if useless:
        print(f"-> Variables inútiles (Ruido) a eliminar en el futuro: {', '.join(useless)}")
    else:
        print("-> Todas las variables están aportando información al modelo.")

def main():
    config = load_config()
    for symbol in config['ingestion']['symbols']:
        run_feature_importance(symbol)

if __name__ == "__main__":
    main()