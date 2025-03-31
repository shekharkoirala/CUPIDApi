from difflib import SequenceMatcher
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
import pickle
from app.utils.config_loader import ConfigLoader
from app.utils.helper import normalize_room_name


def jaccard_similarity_score(s1: str, s2: str) -> float:
    set1, set2 = set(s1.lower().split()), set(s2.lower().split())
    return len(set1 & set2) / len(set1 | set2) if set1 and set2 else 0.0


def is_substring(s1: str, s2: str) -> bool:
    s1_low, s2_low = s1.lower(), s2.lower()
    return (s1_low in s2_low) or (s2_low in s1_low)


def load_vectorizer_pkl_file() -> TfidfVectorizer:
    vectorizer_path = ConfigLoader.get_config()["data"]["vectorizer"]
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer


def numeric_manual_feature(data: list[tuple[str, str, int]]) -> tuple[np.ndarray, np.ndarray]:
    feature_matrix = []
    labels = []
    vectorizer = load_vectorizer_pkl_file()
    for sup_name, ref_name, label in data:
        cos_sim = cosine_similarity(vectorizer.transform([sup_name]), vectorizer.transform([ref_name]))[0, 0]
        jac = jaccard_similarity_score(sup_name, ref_name)
        substr = 1 if is_substring(sup_name, ref_name) else 0
        seq_ratio = SequenceMatcher(None, sup_name.lower(), ref_name.lower()).ratio()
        feature_matrix.append([cos_sim, jac, substr, seq_ratio])
        labels.append(label)

    feature_matrix = standard_scaler(feature_matrix)

    return np.array(feature_matrix), np.array(labels)


def standard_scaler(feature_matrix: np.ndarray) -> np.ndarray:
    config = ConfigLoader.get_config()
    if config["model_configs"]["xgb"]["fixed_params"]["standard_scaler"]:
        scaler = StandardScaler()
        return scaler.fit_transform(feature_matrix)
    else:
        return feature_matrix


# might not use it
def select_k_best(feature_matrix: np.ndarray, labels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    selector = SelectKBest(f_classif, k=k)
    return selector.fit_transform(feature_matrix, labels)


def embedding_feature(data: list[tuple[str, str, int]]) -> tuple[np.ndarray, np.ndarray]:
    model_name = ConfigLoader.get_config()["sentence_transformer_model"]
    model = SentenceTransformer(model_name)
    feature_matrix = []
    labels = []
    for sup_name, ref_name, label in data:
        sup_embedding = model.encode(sup_name)
        ref_embedding = model.encode(ref_name)
        abs_diff = np.abs(sup_embedding - ref_embedding)
        mult_embedding = sup_embedding * ref_embedding
        feature_matrix.append(np.concatenate([sup_embedding, ref_embedding, abs_diff, mult_embedding]))
        labels.append(label)

    feature_matrix = standard_scaler(feature_matrix)

    return np.array(feature_matrix), np.array(labels)


def get_features(
    data: list[tuple[str, str, int]],
) -> tuple[np.ndarray, np.ndarray]:
    try:
        config = ConfigLoader.get_config()
        feature_type = config["feature_type"]
        if feature_type == "numeric_feature":
            return numeric_manual_feature(data)
        elif feature_type == "embedding_feature":
            return embedding_feature(data)
        else:
            raise ValueError(f"Invalid feature type: {feature_type}")
    except Exception as e:
        raise Exception(f"Error getting features: {str(e)}")


def get_feature_inference(ref_name: str, sup_name: str, vectorizer: TfidfVectorizer) -> np.ndarray:
    cos_sim = cosine_similarity(vectorizer.transform([normalize_room_name(sup_name)]), vectorizer.transform([normalize_room_name(ref_name)]))[0, 0]
    jac = jaccard_similarity_score(sup_name, ref_name)
    substr = 1 if is_substring(sup_name, ref_name) else 0
    seq_ratio = SequenceMatcher(None, sup_name.lower(), ref_name.lower()).ratio()
    features = np.array([[cos_sim, jac, substr, seq_ratio]])
    return features
