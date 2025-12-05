import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.model_selection import train_test_split
from sklearn import metrics


class MetamorphicTester:
    """
    Self-contained partition testing utility.
    
    Usage:
        tester = MetamorphicTester("../data/translations.csv")
        tester.run("model_1.onnx")   # or a sklearn model
    """

    # -------------------------------------------------------------
    # INITIALIZER: loads dataset, cleans it, prepares test split
    # -------------------------------------------------------------
    def __init__(self, data_path, seed):
        self.DATA_PATH = data_path
        self.seed = seed
        self.TARGET = "checked"

        # ---------- LOAD & PREPARE DATA ----------
        df_raw = pd.read_csv(self.DATA_PATH, header=None)
        colnames = df_raw.iloc[0].tolist()
        df = pd.read_csv(self.DATA_PATH, skiprows=1, names=colnames)

        df[self.TARGET] = pd.to_numeric(df[self.TARGET], errors="coerce")
        df = df.dropna(subset=[self.TARGET]).copy()
        df[self.TARGET] = df[self.TARGET].astype(int)

        # X, y
        X = df.drop(columns=[self.TARGET]).apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df[self.TARGET]

        # ALWAYS same test set (deterministic)
        _, self.X_test, _, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.seed, stratify=y
        )

    # -------------------------------------------------------------
    # MODEL LOADER (sklearn or ONNX)
    # -------------------------------------------------------------
    def _load_model(self, m):
        if isinstance(m, str):
            # string path → assume ONNX
            return ort.InferenceSession(m, providers=["CPUExecutionProvider"])
        elif isinstance(m, ort.InferenceSession):
            return m
        elif hasattr(m, "predict"):
            return m
        else:
            raise TypeError("Unsupported model type (must be sklearn model or ONNX path).")

    # -------------------------------------------------------------
    # UNIFIED PREDICT FUNCTION
    # -------------------------------------------------------------
    def _predict(self, model, X_part):
        # sklearn
        if hasattr(model, "predict"):
            return model.predict(X_part)

        # ONNX
        elif isinstance(model, ort.InferenceSession):
            input_name = model.get_inputs()[0].name
            X_np = X_part.to_numpy().astype(np.float32)
            outputs = model.run(None, {input_name: X_np})

            # find label output
            label_index = None
            for i, o in enumerate(model.get_outputs()):
                if "label" in o.name.lower():
                    label_index = i
                    break

            if label_index is None:
                raise ValueError("ONNX model missing label output.")

            return np.array(outputs[label_index]).astype(int).flatten()
        
    # -------------------------------------------------------------
    # METAMORPHIC TEST: Gender Flip
    # -------------------------------------------------------------
    def _flip_gender(self, X):
        """
        Returns a gender-flipped copy of X.
        Assumes binary gender encoded as:
            0 = male
            1 = female
        in column 'persoon_geslacht_vrouw'.
        """
        Xf = X.copy()

        def flip(val):
            return 1 - val if val in (0, 1) else val

        Xf['persoon_geslacht_vrouw'] = Xf['persoon_geslacht_vrouw'].apply(flip)
        return Xf
    
    # -------------------------------------------------------------
    # METAMORPHIC TEST: Flip neighbourhoods
    # -------------------------------------------------------------
    def _flip_adres_columns(self, X):
        """
        Flips all columns starting with 'adres_recentste'.
        Assumes binary columns encoded as 0/1.
        """
        Xf = X.copy()

        cols = [c for c in Xf.columns if c.startswith("adres_recentste")]

        for col in cols:
            Xf[col] = Xf[col].apply(lambda v: 1 - v if v in (0, 1) else v)

        return Xf
    
    # -------------------------------------------------------------
    # METAMORPHIC TEST: Add offset to all relationship columns
    # -------------------------------------------------------------
    def _shift_relatie_columns(self, X, delta=1):
        """
        Adds +delta to all integer relatie_* columns.
        """
        Xf = X.copy()
        cols = [c for c in Xf.columns if c.startswith("relatie_")]

        for col in cols:
            Xf[col] = Xf[col] + delta
        
        return Xf
    
    # -------------------------------------------------------------
    # METAMORPHIC TEST: Language Flip
    # -------------------------------------------------------------
    def _flip_language(self, X):
        """
        Flips all columns starting with 'persoonlijke_eigenschappen_nl'.
        Assumes binary columns encoded as 0/1.
        """
        Xf = X.copy()

        cols = [c for c in Xf.columns if c.startswith("persoonlijke_eigenschappen_nl")]

        for col in cols:
            Xf[col] = Xf[col].apply(lambda v: 1 - v if v in (0, 1) else v)

        return Xf


    def compare_changes(self, y_pred_original, y_pred_flipped):
        # Compare predictions
        changes = (y_pred_original != y_pred_flipped)
        n_changed = np.sum(changes)
        frac_changed = n_changed / len(self.X_test)

        print("Accuracy after flip:",
              metrics.accuracy_score(self.y_test, y_pred_flipped))
        print("Number of changed predictions:", n_changed)
        print("Fraction changed: {:.2%}".format(frac_changed))

        # Optional: list indices where predictions changed
        changed_indices = np.where(changes)[0]
        print("Changed indices:", changed_indices.tolist())

    # -------------------------------------------------------------
    # MAIN PUBLIC METHOD
    # -------------------------------------------------------------
    def run(self, model_or_path):
        """
        Runs all metamorphic tests on the given model.
        """
        model = self._load_model(model_or_path)

        print("\n=== Running Metamorphic Tests ===\n")

        # -----------------------------
        # 1. ORIGINAL PREDICTIONS
        # -----------------------------
        y_pred_original = self._predict(model, self.X_test)

        print("Original accuracy:",
              metrics.accuracy_score(self.y_test, y_pred_original))

        # -----------------------------
        # 2. GENDER-FLIP TEST
        # -----------------------------
        print("\n--- Gender Flip Test ---")

        X_flipped_gender = self._flip_gender(self.X_test)
        y_pred_flipped_gender = self._predict(model, X_flipped_gender)

        self.compare_changes(y_pred_original, y_pred_flipped_gender)

        # -----------------------------
        # 2. NEIGHBORHOOD-FLIP TEST
        # -----------------------------
        print("\n--- Neighborhood Flip Test ---")

        X_flipped_neighborhood = self._flip_adres_columns(self.X_test)
        y_pred_flipped_neighborhood = self._predict(model, X_flipped_neighborhood)

        self.compare_changes(y_pred_original, y_pred_flipped_neighborhood)

        # -----------------------------
        # 3. ADD OFFSET RELATIONSHIP TEST
        # -----------------------------
        print("\n--- Offset Relationship Test ---")

        X_relationship = self._flip_adres_columns(self.X_test)
        y_pred_relationship = self._predict(model, X_relationship)

        self.compare_changes(y_pred_original, y_pred_relationship)

        # -----------------------------
        # 4. LANGUAGE FLIP TEST
        # -----------------------------
        print("\n--- Flip Language Test ---")

        X_flipped_language = self._flip_language(self.X_test)
        y_pred_flipped_language = self._predict(model, X_flipped_language)

        self.compare_changes(y_pred_original, y_pred_flipped_language)



