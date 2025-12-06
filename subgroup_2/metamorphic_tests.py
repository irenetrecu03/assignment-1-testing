import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.model_selection import train_test_split

class MetamorphicTester:
    """
    Applies transformations to input data to see if the model's prediction changes.
    Expected behavior: A perfectly fair model should have 0% flip rate for these specific tests.
    """

    def __init__(self, data_path):
        self.DATA_PATH = data_path
        self.TARGET = "checked"

        # ---------- LOAD DATA (Matching your PartitionTester logic) ----------
        # We need the exact same loading logic to ensure columns match
        df_raw = pd.read_csv(self.DATA_PATH, header=None)
        colnames = df_raw.iloc[0].tolist()
        df = pd.read_csv(self.DATA_PATH, skiprows=1, names=colnames)

        df[self.TARGET] = pd.to_numeric(df[self.TARGET], errors="coerce")
        df = df.dropna(subset=[self.TARGET]).copy()
        df[self.TARGET] = df[self.TARGET].astype(int)

        X = df.drop(columns=[self.TARGET]).apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df[self.TARGET]

        # Use the same split logic to keep it consistent
        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Keep a copy of X_test for manipulation
        self.X_base = self.X_test.copy()

    # -------------------------------------------------------------
    # HELPERS (Copied/Adapted from your existing code)
    # -------------------------------------------------------------
    def _load_model(self, m):
        if isinstance(m, str):
            return ort.InferenceSession(m, providers=["CPUExecutionProvider"])
        elif isinstance(m, ort.InferenceSession):
            return m
        elif hasattr(m, "predict"):
            return m
        else:
            raise TypeError("Unsupported model type.")

    def _predict(self, model, X_part):
        # SKLEARN
        if hasattr(model, "predict"):
            return model.predict(X_part)
        
        # ONNX
        elif isinstance(model, ort.InferenceSession):
            input_name = model.get_inputs()[0].name
            X_np = X_part.to_numpy().astype(np.float32)
            outputs = model.run(None, {input_name: X_np})
            
            # Find label output
            label_index = None
            for i, o in enumerate(model.get_outputs()):
                if "label" in o.name.lower():
                    label_index = i
                    break
            if label_index is None: 
                # Fallback for some sklearn-onnx conversions where output 0 is label
                label_index = 0 
                
            return np.array(outputs[label_index]).astype(int).flatten()

    def _calculate_violations(self, name, original_preds, new_preds):
        """
        Calculates how many times the prediction FLIPPED.
        """
        # A violation is when Pred_orig != Pred_new
        violations = np.sum(original_preds != new_preds)
        total = len(original_preds)
        rate = (violations / total) * 100
        
        print(f"Test: {name}")
        print(f"-----------------------------------")
        print(f"Total rows tested: {total}")
        print(f"Prediction Flips:  {violations}")
        print(f"Violation Rate:    {rate:.2f}%")
        print(f"-----------------------------------\n")
        return rate

    # -------------------------------------------------------------
    # THE METAMORPHIC TRANSFORMATIONS
    # -------------------------------------------------------------
    
    def test_gender_flip(self, model):
        """
        MR: In a fair system, changing gender (Man <-> Woman) should NOT change fraud risk.
        Target Feature: persoon_geslacht_vrouw
        """
        print(">>> Running MR: Gender Flip")
        
        # 1. Get Baseline Predictions
        preds_base = self._predict(model, self.X_base)
        
        # 2. Apply Transformation
        X_mutant = self.X_base.copy()
        # Flip 0 to 1, and 1 to 0
        X_mutant['persoon_geslacht_vrouw'] = 1 - X_mutant['persoon_geslacht_vrouw']
        
        # 3. Get Mutant Predictions
        preds_mutant = self._predict(model, X_mutant)
        
        # 4. Compare
        self._calculate_violations("Gender Flip", preds_base, preds_mutant)

    def test_language_flip(self, model):
        """
        MR: In a fair system, understanding Dutch vs Not Understanding 
        should not be the sole decider of fraud.
        Target Feature: persoonlijke_eigenschappen_nl_begrijpen3 (or similar)
        """
        target_col = 'persoonlijke_eigenschappen_nl_begrijpen3'
        
        if target_col not in self.X_base.columns:
            print(f"Skipping Language Flip: {target_col} not found.")
            return

        print(">>> Running MR: Language Proficiency Flip")
        
        preds_base = self._predict(model, self.X_base)
        
        X_mutant = self.X_base.copy()
        X_mutant[target_col] = 1 - X_mutant[target_col]
        
        preds_mutant = self._predict(model, X_mutant)
        self._calculate_violations("Language Proficiency Flip", preds_base, preds_mutant)

    def test_neighborhood_shuffle(self, model):
        """
        MR: Changing the neighborhood (while keeping financial/personal status same)
        should not change fraud risk.
        We swap everyone living in 'Feijenoord' to 'Kralingen' (or generic swap).
        """
        # Let's find two neighborhoods that exist in the columns
        n1 = 'adres_recentste_wijk_feijenoord'
        n2 = 'adres_recentste_wijk_kralingen_c'
        
        if n1 not in self.X_base.columns or n2 not in self.X_base.columns:
            print("Skipping Neighborhood Shuffle: Columns not found.")
            return

        print(">>> Running MR: Neighborhood Swap (Feijenoord <-> Kralingen)")
        
        # Filter to only people who live in one of these two places to make the swap logical
        mask = (self.X_base[n1] == 1) | (self.X_base[n2] == 1)
        X_subset = self.X_base[mask].copy()
        
        if len(X_subset) == 0:
            print("No relevant rows for neighborhood swap.")
            return

        preds_base = self._predict(model, X_subset)
        
        # Apply Swap
        # If they were in n1, move to n2. If in n2, move to n1.
        X_mutant = X_subset.copy()
        X_mutant[n1] = X_subset[n2]
        X_mutant[n2] = X_subset[n1]
        
        preds_mutant = self._predict(model, X_mutant)
        self._calculate_violations("Neighborhood Swap", preds_base, preds_mutant)

    # -------------------------------------------------------------
    # MAIN RUNNER
    # -------------------------------------------------------------
    def run(self, model_or_path):
        model = self._load_model(model_or_path)
        print("\n=========================================")
        print("      METAMORPHIC TEST RESULTS")
        print("=========================================\n")
        
        self.test_gender_flip(model)
        self.test_language_flip(model)
        self.test_neighborhood_shuffle(model)