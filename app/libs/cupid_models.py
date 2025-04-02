import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
import seaborn as sns
import matplotlib.pyplot as plt
from app.utils.helper import get_next_experiment_folder
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Optional


class XGBoostModel:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.model_path = self.config["model_configs"]["xgb"]["model_path"]
        model_dir = os.path.dirname(self.model_path)
        os.makedirs(self.config["model_configs"]["xgb"]["report_path"], exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        self.model: xgb.XGBClassifier | None = None

    def load_model(self):
        # check if model exists
        if not os.path.exists(self.model_path):
            self.logger.debug(f"Model file not found at {self.model_path}. Trigger /trainModel endpoint to train model.")
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Trigger /trainModel endpoint to train model.")
        else:
            self.model = xgb.XGBClassifier(
                use_label_encoder=False,
                eval_metric=self.config["model_configs"]["xgb"]["fixed_params"]["eval_metric"],
                random_state=self.config["random_state"],
            )
            self.model.load_model(self.model_path)

    def train_model(self, X_train, X_test, y_train, y_test):
        self.model = xgb.XGBClassifier(
            use_label_encoder=False,
            eval_metric=self.config["model_configs"]["xgb"]["fixed_params"]["eval_metric"],
            random_state=self.config["random_state"],
        )

        if self.config["model_configs"]["xgb"]["fixed_params"]["apply_imbalance"]:
            imbalance_ratio = sum(y_train == 0) / sum(y_train == 1)
            self.model.set_params(scale_pos_weight=imbalance_ratio)

        self.model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
        self.model.save_model(self.model_path)
        y_pred = self.model.predict(X_test)

        exp_folder = get_next_experiment_folder(self.config["model_configs"]["xgb"]["report_path"])

        with open(os.path.join(exp_folder, "classification_report.md"), "w") as f:
            f.write(classification_report(y_test, y_pred))

        with open(os.path.join(exp_folder, "accuracy_score.txt"), "w") as f:
            f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}")

        cm_path = os.path.join(exp_folder, "confusion_matrix.png")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Match", "Match"], yticklabels=["Not Match", "Match"])
        plt.savefig(cm_path)

        feature_importance_path = os.path.join(exp_folder, "feature_importance.png")

        if self.config["feature_type"] == "numeric_feature":
            feature_names = [
                "cosine_similarity",
                "jaccard_similarity",
                "substring",
                "sequence_ratio",
                "embedding_cosine_similarity",
                "char_ngram_jaccard_score",
            ]
            plt.figure(figsize=(10, 7))
            plt.barh(feature_names, self.model.feature_importances_)
            plt.savefig(feature_importance_path)
        # else:
        #     # not make sense for embedding feature
        #     pass


class ModelManager:
    _instance: Optional[XGBoostModel] = None

    @classmethod
    def get_model(cls) -> XGBoostModel:
        if cls._instance is None:
            raise RuntimeError("Model not initialized")
        return cls._instance

    @classmethod
    def set_model(cls, model: XGBoostModel) -> None:
        cls._instance = model


class VectorizerManager:
    _instance: Optional[TfidfVectorizer] = None

    @classmethod
    def get_vectorizer(cls) -> TfidfVectorizer:
        if cls._instance is None:
            raise RuntimeError("Vectorizer not initialized")
        return cls._instance

    @classmethod
    def set_vectorizer(cls, vectorizer: TfidfVectorizer) -> None:
        cls._instance = vectorizer
