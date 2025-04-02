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
import traceback


def jaccard_similarity_score(s1_tokens: set, s2_tokens: set) -> float:
    if not s1_tokens and not s2_tokens:
        return 1.0  # define Jaccard(∅, ∅) = 1 (both empty strings considered identical)
    if not s1_tokens or not s2_tokens:
        return 0.0
    inter = s1_tokens.intersection(s2_tokens)
    union = s1_tokens.union(s2_tokens)
    return float(len(inter) / len(union))


def is_substring(s1: str, s2: str) -> bool:
    s1_low, s2_low = s1.lower(), s2.lower()
    return (s1_low in s2_low) or (s2_low in s1_low)


def load_vectorizer_pkl_file() -> TfidfVectorizer:
    vectorizer_path = ConfigLoader.get_config()["data"]["vectorizer"]
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)
    return vectorizer


def embedding_cosine_similarity(s1: str, s2: str, embedding_model: SentenceTransformer) -> float:
    emb1 = embedding_model.encode([s1])[0]
    emb2 = embedding_model.encode([s2])[0]
    # Compute cosine similarity manually
    # (avoid zero-division by adding a small epsilon to norms)
    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    cos_sim = np.dot(emb1, emb2) / (norm1 * norm2)
    return float(cos_sim)


def char_ngram_jaccard(s1: str, s2: str, n: int = 3) -> float:
    """
    Compute Jaccard similarity of character n-grams between two strings.
    By default, uses n=3 (trigrams).
    """
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0

    # Generate set of character n-grams for each string
    def ngrams(text, n):
        return {text[i : i + n] for i in range(len(text) - n + 1)}  # set of all n-length substrings

    set1 = ngrams(s1, n)
    set2 = ngrams(s2, n)
    return jaccard_similarity_score(set1, set2)


def numeric_manual_feature(data: list[tuple[str, str, int]]) -> tuple[np.ndarray, np.ndarray]:
    try:
        feature_matrix = []
        labels = []
        vectorizer = load_vectorizer_pkl_file()
        model_name = ConfigLoader.get_config()["sentence_transformer_model"]
        embedding_model = SentenceTransformer(model_name)
        for sup_name, ref_name, label in data:
            ref_norm = normalize_room_name(ref_name)
            sup_norm = normalize_room_name(sup_name)
            ref_tokens = set(ref_norm.split()) if ref_norm else set()
            sup_tokens = set(sup_norm.split()) if sup_norm else set()

            cos_sim = cosine_similarity(vectorizer.transform([sup_norm]), vectorizer.transform([ref_norm]))[0, 0]
            jac = jaccard_similarity_score(sup_tokens, ref_tokens)
            substr = 1 if is_substring(sup_norm, ref_norm) else 0
            seq_ratio = SequenceMatcher(None, sup_norm.lower(), ref_norm.lower()).ratio()
            emb_cos_sim = embedding_cosine_similarity(sup_norm.lower(), ref_norm.lower(), embedding_model)
            char_ngram_jaccard_score = char_ngram_jaccard(sup_norm.lower(), ref_norm.lower())
            feature_matrix.append([0.8 * cos_sim, jac, substr, seq_ratio, 1.5 * emb_cos_sim, char_ngram_jaccard_score])
            labels.append(label)

        feature_matrix = standard_scaler(feature_matrix)

        return np.array(feature_matrix), np.array(labels)
    except Exception as e:
        traceback.print_exc()
        raise Exception(f"Error getting numeric manual features: {str(e)}")


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
    feature_type = ConfigLoader.get_config()["feature_type"]
    model_name = ConfigLoader.get_config()["sentence_transformer_model"]
    embedding_model = SentenceTransformer(model_name)
    if feature_type == "numeric_feature":
        ref_norm = normalize_room_name(ref_name)
        sup_norm = normalize_room_name(sup_name)

        ref_tokens: set[str] = set(ref_norm.split()) if ref_norm else set()
        sup_tokens: set[str] = set(sup_norm.split()) if sup_norm else set()

        cos_sim = cosine_similarity(vectorizer.transform([sup_norm]), vectorizer.transform([ref_norm]))[0, 0]
        jac = jaccard_similarity_score(sup_tokens, ref_tokens)
        substr = 1 if is_substring(sup_norm, ref_norm) else 0
        seq_ratio = SequenceMatcher(None, sup_norm.lower(), ref_norm.lower()).ratio()
        emb_cos_sim = embedding_cosine_similarity(sup_norm.lower(), ref_norm.lower(), embedding_model)
        char_ngram_jaccard_score = char_ngram_jaccard(sup_norm.lower(), ref_norm.lower())
        features = np.array([[cos_sim, jac, substr, seq_ratio, emb_cos_sim, char_ngram_jaccard_score]])
        return features

    elif feature_type == "embedding_feature":
        model_name = ConfigLoader.get_config()["sentence_transformer_model"]
        model = SentenceTransformer(model_name)
        sup_embedding = model.encode(normalize_room_name(sup_name))
        ref_embedding = model.encode(normalize_room_name(ref_name))
        abs_diff = np.abs(sup_embedding - ref_embedding)
        mult_embedding = sup_embedding * ref_embedding
        features = np.concatenate([sup_embedding, ref_embedding, abs_diff, mult_embedding])
        return features
