import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.model_selection import train_test_split


class PartitionTester:
    """
    Self-contained partition testing utility.
    
    Usage:
        tester = PartitionTester("../data/translations.csv")
        tester.run("model_1.onnx")   # or a sklearn model
    """

    # -------------------------------------------------------------
    # INITIALIZER: loads dataset, cleans it, prepares test split
    # -------------------------------------------------------------
    def __init__(self, data_path):
        self.DATA_PATH = data_path
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
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # ---------- DEFINE PARTITIONS ----------
        self.partitions = [
            # Gender-based partitions
            {"name": "men", "condition": lambda df: df['persoon_geslacht_vrouw'] == 0},
            {"name": "women", "condition": lambda df: df['persoon_geslacht_vrouw'] == 1},
            # Age-based partitions
            {"name": "young_adults", "condition": lambda df: df['persoon_leeftijd_bij_onderzoek'] < 30},
            {"name": "middle_aged", "condition": lambda df: (df['persoon_leeftijd_bij_onderzoek'] >= 30) & (df['persoon_leeftijd_bij_onderzoek'] < 60)},
            {"name": "seniors", "condition": lambda df: df['persoon_leeftijd_bij_onderzoek'] >= 60},
            # Family status
            {"name": "single_parents", "condition": lambda df: (df['relatie_kind_heeft_kinderen'] == 1) & (df['relatie_partner_huidige_partner___partner__gehuwd_'] == 0)},
            {"name": "married_with_children", "condition": lambda df: (df['relatie_kind_heeft_kinderen'] == 1) & (df['relatie_partner_huidige_partner___partner__gehuwd_'] == 1)},
            {"name": "no_children_no_partner", "condition": lambda df: (df['relatie_kind_heeft_kinderen'] == 0) & (df['relatie_partner_huidige_partner___partner__gehuwd_'] == 0)},
            # Marital status
            {"name": "currently_married", "condition": lambda df: df['relatie_partner_huidige_partner___partner__gehuwd_'] == 1},
            {"name": "currently_unmarried_with_partner", "condition": lambda df: df['relatie_partner_aantal_partner___partner__ongehuwd_'] > 0},
            {"name": "currently_single", "condition": lambda df: (
                (df['relatie_partner_huidige_partner___partner__gehuwd_'] == 0) & 
                (df['relatie_partner_aantal_partner___partner__ongehuwd_'] == 0)
            )},
            {"name": "multiple_unmarried_partners", "condition": lambda df: df['relatie_partner_aantal_partner___partner__ongehuwd_'] > 1},
            {"name": "likely_divorced", "condition": lambda df: (
                (df['relatie_partner_aantal_partner___partner__gehuwd_'] > 0) &  # Had married partner historically
                (df['relatie_partner_huidige_partner___partner__gehuwd_'] == 0)  # Not currently married
            )},
            {"name": "likely_divorced_with_children", "condition": lambda df: (
                (df['relatie_partner_aantal_partner___partner__gehuwd_'] > 0) &
                (df['relatie_partner_huidige_partner___partner__gehuwd_'] == 0) &
                (df['relatie_kind_heeft_kinderen'] == 1)
            )},
            {"name": "likely_divorced_no_children", "condition": lambda df: (
                (df['relatie_partner_aantal_partner___partner__gehuwd_'] > 0) &
                (df['relatie_partner_huidige_partner___partner__gehuwd_'] == 0) &
                (df['relatie_kind_heeft_kinderen'] == 0)
            )},
            {"name": "divorced_women", "condition": lambda df: (
                (df['relatie_partner_aantal_partner___partner__gehuwd_'] > 0) &
                (df['relatie_partner_huidige_partner___partner__gehuwd_'] == 0) &
                (df['persoon_geslacht_vrouw'] == 1)
            )},
            {"name": "divorced_women_with_children", "condition": lambda df: (
                (df['relatie_partner_aantal_partner___partner__gehuwd_'] > 0) &
                (df['relatie_partner_huidige_partner___partner__gehuwd_'] == 0) &
                (df['persoon_geslacht_vrouw'] == 1) &
                (df['relatie_kind_heeft_kinderen'] == 1)
            )},
            # Currently cohabiting but not married
            {"name": "cohabiting_unmarried", "condition": lambda df: (
                (df['relatie_partner_aantal_partner___partner__ongehuwd_'] > 0) &
                (df['relatie_partner_huidige_partner___partner__gehuwd_'] == 0) &
                (df['relatie_overig_kostendeler'] == 1)  # Cost-sharer = living together
            )},
            # Dutch understanding
            {"name": "understands_dutch", "condition": lambda df: df['persoonlijke_eigenschappen_nl_begrijpen3'] == 1},
            {"name": "does_not_understand_dutch", "condition": lambda df: df['persoonlijke_eigenschappen_nl_begrijpen3'] == 0},
            # Short time at address + language issues (recent immigrants)
            {"name": "likely_recent_arrival_non_Dutch", "condition": lambda df: (
                (df['adres_dagen_op_adres'] < 365) & 
                (df['adres_recentste_plaats_rotterdam'] == 1) &
                (df['persoonlijke_eigenschappen_nl_begrijpen3'] == 0)
            )},
            {"name": "likely_recent_arrival_Dutch", "condition": lambda df: (
                (df['adres_dagen_op_adres'] < 365) & 
                (df['adres_recentste_plaats_rotterdam'] == 1) &
                (df['persoonlijke_eigenschappen_nl_begrijpen3'] == 1)
            )},
            {"name": "less_established_residents_non_Dutch", "condition": lambda df: (
                (df['adres_dagen_op_adres'] < 1825) &
                (df['adres_dagen_op_adres'] >= 365) &
                (df['adres_recentste_plaats_rotterdam'] == 1) &
                (df['persoonlijke_eigenschappen_nl_begrijpen3'] == 0)
            )},
            {"name": "less_established_residents_Dutch", "condition": lambda df: (
                (df['adres_dagen_op_adres'] < 1825) &
                (df['adres_dagen_op_adres'] >= 365) &
                (df['adres_recentste_plaats_rotterdam'] == 1) &
                (df['persoonlijke_eigenschappen_nl_begrijpen3'] == 1)
            )},
            {"name": "established_residents_non_Dutch", "condition": lambda df: (
                (df['adres_dagen_op_adres'] > 1825) &  # 5+ years
                (df['adres_recentste_plaats_rotterdam'] == 1) &
                (df['persoonlijke_eigenschappen_nl_begrijpen3'] == 0)
            )},
            {"name": "established_residents_Dutch", "condition": lambda df: (
                (df['adres_dagen_op_adres'] > 1825) &  # 5+ years
                (df['adres_recentste_plaats_rotterdam'] == 1) &
                (df['persoonlijke_eigenschappen_nl_begrijpen3'] == 1)
            )},
            # Most recent borough
            {"name": "charlois", "condition": lambda df: df['adres_recentste_wijk_charlois'] == 1},
            {"name": "delfshaven", "condition": lambda df: df['adres_recentste_wijk_delfshaven'] == 1},
            {"name": "feijenoord", "condition": lambda df: df['adres_recentste_wijk_feijenoord'] == 1},
            {"name": "ijsselmonde", "condition": lambda df: df['adres_recentste_wijk_ijsselmonde'] == 1},
            {"name": "kralingen_c", "condition": lambda df: df['adres_recentste_wijk_kralingen_c'] == 1},
            {"name": "noord", "condition": lambda df: df['adres_recentste_wijk_noord'] == 1},
            {"name": "prins_alexa", "condition": lambda df: df['adres_recentste_wijk_prins_alexa'] == 1},
            {"name": "stadscentru", "condition": lambda df: df['adres_recentste_wijk_stadscentru'] == 1},
            # Obstacles
            {"name": "psychological_obstacles", "condition": lambda df: df['belemmering_psychische_problemen'] == 1},
            {"name": "no_psychological_obstacles", "condition": lambda df: df['belemmering_psychische_problemen'] == 0},
            {"name": "living_situation_obstacles", "condition": lambda df: df['belemmering_woonsituatie'] == 1},
            {"name": "no_living_situation_obstacles", "condition": lambda df: df['belemmering_woonsituatie'] == 0},
            {"name": "financial_obstacles", "condition": lambda df: df['belemmering_financiele_problemen'] == 1},
            {"name": "no_financial_obstacles", "condition": lambda df: df['belemmering_financiele_problemen'] == 0},
            # Multiple obstacles
            {"name": "psychological_financial_obstacles", "condition": lambda df: (
                (df['belemmering_psychische_problemen'] == 1) & 
                (df['belemmering_financiele_problemen'] == 1)
            )},
            {"name": "psychological_financial_living_obstacles", "condition": lambda df: (
                (df['belemmering_psychische_problemen'] == 1) & 
                (df['belemmering_financiele_problemen'] == 1) &
                (df['belemmering_woonsituatie'] == 1)
            )},
            {"name": "no_obstacles", "condition": lambda df: (
                (df['belemmering_psychische_problemen'] == 0) & 
                (df['belemmering_financiele_problemen'] == 0) &
                (df['belemmering_woonsituatie'] == 0)
            )},
        ]

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
    # MAIN PUBLIC METHOD
    # -------------------------------------------------------------
    def run(self, model_or_path):
        """
        Execute full partition tests on the given model.
        """
        model = self._load_model(model_or_path)

        print("\n=========================================")
        print("      PARTITION TEST RESULTS")
        print("=========================================\n")

        for part in self.partitions:
            name = part["name"]
            cond = part["condition"]

            df_part = self.X_test[cond(self.X_test)]
            idx = df_part.index
            true_labels = self.y_test.loc[idx].astype(int)

            if df_part.empty:
                print(f"Skipping {name}: no rows.\n")
                continue

            preds = self._predict(model, df_part)

            TP = np.sum((preds == 1) & (true_labels == 1))
            TN = np.sum((preds == 0) & (true_labels == 0))
            FP = np.sum((preds == 1) & (true_labels == 0))
            FN = np.sum((preds == 0) & (true_labels == 1))

            N = len(df_part)
            fraud_rate = true_labels.mean()
            pred_rate = preds.mean()
            acc = (TP + TN) / N
            FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
            FNR = FN / (FN + TP) if (FN + TP) > 0 else 0
            TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
            TNR = TN / (TN + FP) if (TN + FP) > 0 else 0

            print("===================================")
            print(f"Partition: {name}")
            print("===================================")
            print(f"Data points: {N}")
            print(f"Actual fraud rate:   {fraud_rate*100:.2f}%")
            print(f"Predicted fraud rate:{pred_rate*100:.2f}%")
            print("\n--- Confusion Matrix ---")
            print(f"TP={TP}  TN={TN}  FP={FP}  FN={FN}")
            print("\n--- Metrics ---")
            print(f"Accuracy: {acc*100:.2f}%")
            print(f"FPR: {FPR*100:.2f}%")
            print(f"FNR: {FNR*100:.2f}%")
            print(f"TPR/Recall: {TPR*100:.2f}%")
            print(f"TNR: {TNR*100:.2f}%\n")
