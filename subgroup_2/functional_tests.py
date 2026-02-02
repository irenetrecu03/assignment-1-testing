import pandas as pd
import numpy as np
import onnxruntime as ort
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

class FunctionalTester:
    def __init__(self, data_path):
        self.DATA_PATH = data_path
        self.TARGET = "checked"
        
        # Load Data
        df_header = pd.read_csv(self.DATA_PATH, nrows=0)
        colnames = df_header.columns.tolist()
        df = pd.read_csv(self.DATA_PATH, skiprows=1, names=colnames, low_memory=False)

        df[self.TARGET] = pd.to_numeric(df[self.TARGET], errors="coerce")
        df = df.dropna(subset=[self.TARGET]).copy()
        df[self.TARGET] = df[self.TARGET].astype(int)

        X = df.drop(columns=[self.TARGET]).apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df[self.TARGET]

        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

    def _load_model(self, m):
        if isinstance(m, str): return ort.InferenceSession(m, providers=["CPUExecutionProvider"])
        return m

    def run(self, model_path):
        model = self._load_model(model_path)
        print(f"\n=== FUNCTIONAL TEST: {model_path} ===")
        
        input_name = model.get_inputs()[0].name
        X_np = self.X_test.to_numpy().astype(np.float32)
        
        # Get Predictions
        outputs = model.run(None, {input_name: X_np})
        
        # Label is usually index 0
        preds = np.array(outputs[0]).astype(int).flatten()
        
        # Try to get probabilities for AUC (usually index 1)
        try:
            probs_list = outputs[1]
            # specific logic for sklearn-onnx maps {0: p0, 1: p1}
            y_proba = np.array([p[1] for p in probs_list])
            auc = roc_auc_score(self.y_test, y_proba)
        except:
            auc = 0.5 # Fallback if probas missing
            
        # Metrics
        acc = accuracy_score(self.y_test, preds)
        tn, fp, fn, tp = confusion_matrix(self.y_test, preds).ravel()
        
        print(f"Accuracy: {acc:.4f}")
        print(f"AUC:      {auc:.4f}")
        print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # The key differentiator: False Positive Rate
        fpr = fp / (fp + tn) * 100
        print(f"False Positive Rate (Innocents Flagged): {fpr:.2f}%")
        print("============================================\n")