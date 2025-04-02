from typing import Dict, Any, List
from app.utils.config_loader import ConfigLoader
from app.utils.logger import CustomLogger
import polars as pl
import traceback
from app.utils.helper import normalize_room_name
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import time
import pickle
import os
import spacy
from collections import defaultdict


class DataProcessor:
    # Room type hierarchy
    ROOM_HIERARCHY = {
        "suite": [
            "suite",
            "sweet",
            "suit",
            "presidential",
            "royal",
            "executive",
            "junior",
            "deluxe",
            "luxury",
            "premium",
            "signature",
            "master",
            "vip",
        ],
        "apartment": [
            "apartment",
            "apt",
            "flat",
            "residence",
            "condo",
            "villa",
            "house",
            "bungalow",
            "cottage",
            "chalet",
            "studio",
            "loft",
            "maisonette",
            "townhouse",
        ],
        "room": ["room", "chamber", "bedroom", "guest room", "standard", "superior", "comfort", "basic", "economy", "standard", "deluxe", "premium"],
        "dormitory": ["dormitory", "dorm", "hostel", "shared", "mixed dorm", "bunk"],
    }

    # Room features to extract
    ROOM_FEATURES = {
        "bed_type": ["single", "double", "twin", "triple", "queen", "king", "sofa bed", "bunk bed", "extra bed", "rollaway"],
        "view_type": [
            "ocean",
            "garden",
            "city",
            "mountain",
            "pool",
            "sea",
            "lake",
            "river",
            "beach",
            "golf",
            "courtyard",
            "balcony",
            "terrace",
            "patio",
        ],
        "room_size": ["small", "medium", "large", "spacious", "xxl", "junior", "standard", "deluxe"],
        "amenities": [
            "jacuzzi",
            "balcony",
            "kitchen",
            "living room",
            "bathroom",
            "private bathroom",
            "shared bathroom",
            "bathtub",
            "shower",
            "refrigerator",
            "microwave",
            "air conditioning",
            "heating",
            "tv",
            "wifi",
            "internet",
            "safe",
            "desk",
            "closet",
            "wardrobe",
            "minibar",
            "coffee maker",
            "tea maker",
        ],
        "accessibility": ["accessible", "wheelchair", "mobility", "hearing", "visual", "rollin shower", "grab bars"],
        "smoking": ["smoking", "non smoking", "no smoking", "smoking allowed", "smoking not allowed", "smoking prohibited"],
        "pet_policy": ["pet friendly", "no pets", "no dogs", "dogs allowed", "pets allowed", "pet free"],
        "occupancy": ["single use", "double occupancy", "triple occupancy", "quadruple", "family", "adults only", "children allowed"],
        "location": ["ground floor", "upper floor", "top floor", "lower level", "upper level", "tower", "annex", "main building"],
        "style": ["japanese", "western", "traditional", "modern", "classic", "contemporary", "luxury", "budget", "eco"],
        "bedroom_count": [
            "1 bedroom",
            "2 bedroom",
            "3 bedroom",
            "4 bedroom",
            "5 bedroom",
            "6 bedroom",
            "7 bedroom",
            "8 bedroom",
            "9 bedroom",
            "10 bedroom",
        ],
    }

    # Abbreviations mapping
    ABBREVIATIONS = {
        "r.o.": "room",
        "std": "standard",
        "dbl": "double",
        "sgl": "single",
        "tpl": "triple",
        "qdr": "quadruple",
        "apt": "apartment",
        "rm": "room",
        "br": "bedroom",
        "bth": "bathroom",
        "wifi": "wireless internet",
        "ac": "air conditioning",
    }

    @staticmethod
    def normalize_abbreviations(text: str) -> str:
        """Normalize common abbreviations in room names."""
        text = text.lower()
        for abbr, full in DataProcessor.ABBREVIATIONS.items():
            text = text.replace(abbr, full)
        return text

    @staticmethod
    def extract_bedroom_count(room_name: str) -> int:
        """Extract the number of bedrooms from room name."""
        room_name = room_name.lower()
        for count in range(1, 11):
            if f"{count} bedroom" in room_name:
                return count
        return 1  # Default to 1 if not specified

    @staticmethod
    def extract_features(room_name: str) -> Dict[str, List[str]]:
        """Extract room features from the room name."""
        features = defaultdict(list)
        room_name = DataProcessor.normalize_abbreviations(room_name)
        words = set(room_name.lower().split())  # Use set for faster lookups

        # Pre-compute bedroom count
        bedroom_count = DataProcessor.extract_bedroom_count(room_name)
        features["bedroom_count"] = [f"{bedroom_count} bedroom"]

        # Process all features in one pass
        for feature_type, keywords in DataProcessor.ROOM_FEATURES.items():
            if feature_type == "bedroom_count":
                continue
            # Use set intersection for faster matching
            matched = words.intersection(keywords)
            if matched:
                features[feature_type] = list(matched)

        return dict(features)

    @staticmethod
    def get_room_type(room_name: str) -> str:
        """Determine the base room type from the hierarchy."""
        room_name = room_name.lower()
        for base_type, variations in DataProcessor.ROOM_HIERARCHY.items():
            if any(var in room_name for var in variations):
                return base_type
        return "room"  # default

    @staticmethod
    def calculate_semantic_similarity(str1: str, str2: str, nlp) -> float:
        """Calculate semantic similarity using spaCy."""
        doc1 = nlp(str1)
        doc2 = nlp(str2)
        return doc1.similarity(doc2)

    @staticmethod
    def features_match(features1: Dict[str, List[str]], features2: Dict[str, List[str]]) -> bool:
        """Check if room features are compatible with strict rules."""
        # Quick check for bedroom count
        if features1["bedroom_count"] != features2["bedroom_count"]:
            return False

        # Use sets for faster matching
        for feature_type in ["smoking", "pet_policy", "bed_type"]:
            if feature_type in features1 and feature_type in features2:
                if features1[feature_type] and features2[feature_type]:
                    if not set(features1[feature_type]).intersection(features2[feature_type]):
                        return False

        # Process other features
        for feature_type in DataProcessor.ROOM_FEATURES:
            if feature_type not in ["bedroom_count", "smoking", "pet_policy", "bed_type"]:
                if feature_type in features1 and feature_type in features2:
                    if features1[feature_type] and features2[feature_type]:
                        if not set(features1[feature_type]).intersection(features2[feature_type]):
                            return False

        return True

    @staticmethod
    def process_hotel_rooms(args):
        """Process all rooms for a single hotel."""
        ref_data, sup_data, vectorizer, room_docs, lp_id = args

        # Prepare room names and pre-compute features
        ref_names = [r["normalized_room_name"] for r in ref_data]
        sup_names = [s["normalized_supplier_room_name"] for s in sup_data]

        # Pre-compute features for all rooms
        ref_features = {name: DataProcessor.extract_features(name) for name in ref_names}
        sup_features = {name: DataProcessor.extract_features(name) for name in sup_names}

        # Pre-compute room types
        ref_types = {name: DataProcessor.get_room_type(name) for name in ref_names}
        sup_types = {name: DataProcessor.get_room_type(name) for name in sup_names}

        # Transform room names to vectors for this hotel only
        ref_vecs = vectorizer.transform(ref_names)
        sup_vecs = vectorizer.transform(sup_names)

        hotel_matches = set()
        hotel_not_matches = set()

        # Process each supplier room
        for i, s_name in enumerate(sup_names):
            best_match = None
            best_score = 0

            # Get pre-computed data
            s_doc = room_docs[s_name]
            s_features = sup_features[s_name]
            s_type = sup_types[s_name]

            # Extract numbers and block/floor information
            s_numbers = [int(s) for s in s_name.split() if s.isdigit()]
            s_blocks = [word for word in s_name.split() if word.lower() in ["bloque", "block", "floor", "level", "tower"]]

            # Calculate cosine similarity for this supplier room against all reference rooms
            s_vec = sup_vecs[i : i + 1]
            pair_sims = cosine_similarity(s_vec, ref_vecs)[0]

            # Get indices of top 2 most similar reference rooms
            top_indices = pair_sims.argsort()[-2:][::-1]

            # Process only the top 2 reference rooms from this hotel
            for j in top_indices:
                r_name = ref_names[j]
                sim_score = pair_sims[j]

                # Skip if similarity is too low
                if sim_score < 0.5:
                    continue

                # Extract numbers and block/floor information from reference room
                r_numbers = [int(s) for s in r_name.split() if s.isdigit()]
                r_blocks = [word for word in r_name.split() if word.lower() in ["bloque", "block", "floor", "level", "tower"]]

                # Check for number mismatches
                if s_numbers and r_numbers:
                    if s_numbers != r_numbers:
                        # If numbers are different, reduce similarity score significantly
                        sim_score *= 0.5

                # Check for block/floor mismatches
                if s_blocks or r_blocks:
                    if s_blocks != r_blocks:
                        # If blocks/floors are different, reduce similarity score significantly
                        sim_score *= 0.5

                # Get pre-computed data
                r_doc = room_docs[r_name]
                r_features = ref_features[r_name]
                r_type = ref_types[r_name]

                # Quick check for bedroom count mismatch
                if s_features["bedroom_count"] != r_features["bedroom_count"]:
                    continue

                # Strict bed type matching
                s_bed_types = set(s_features.get("bed_type", []))
                r_bed_types = set(r_features.get("bed_type", []))

                # Check for incompatible bed types
                incompatible_pairs = {
                    ("single", "double"),
                    ("double", "single"),
                    ("single", "twin"),
                    ("twin", "single"),
                    ("single", "queen"),
                    ("queen", "single"),
                    ("single", "king"),
                    ("king", "single"),
                    ("double", "twin"),
                    ("twin", "double"),
                    ("double", "queen"),
                    ("queen", "double"),
                    ("double", "king"),
                    ("king", "double"),
                }

                has_incompatible = any(
                    (s_type, r_type) in incompatible_pairs or (r_type, s_type) in incompatible_pairs
                    for s_type in s_bed_types
                    for r_type in r_bed_types
                )

                if has_incompatible:
                    # If bed types are incompatible, reduce similarity score significantly
                    sim_score *= 0.3

                # Calculate semantic similarity
                semantic_score = s_doc.similarity(r_doc)

                # Calculate type compatibility
                type_compatible = s_type == r_type or (
                    (s_type == "suite" and r_type in ["apartment", "room"]) or (s_type == "apartment" and r_type == "room")
                )

                # Calculate feature compatibility
                features_compatible = DataProcessor.features_match(s_features, r_features)

                # Combined score with weights
                combined_score = (
                    0.4 * sim_score  # Use adjusted cosine similarity
                    + 0.4 * semantic_score
                    + 0.1 * float(type_compatible)
                    + 0.1 * float(features_compatible)
                )

                if combined_score > best_score:
                    best_score = combined_score
                    best_match = (s_name, r_name, combined_score)

            if best_match and best_match[2] >= 0.6:
                # Add lp_id to the match tuple
                hotel_matches.add(best_match + (lp_id,))
            else:
                hotel_not_matches.add((s_name, best_match[1] if best_match else None, 0, lp_id))

        return hotel_matches, hotel_not_matches

    @staticmethod
    async def process_data(force_update: bool) -> Dict[str, Any]:
        logger = CustomLogger.get_logger()
        config = ConfigLoader.get_config()

        try:
            if not force_update:
                if os.path.exists(config["data"]["processed_data"]):
                    logger.info("Processed data already exists, skipping processing")
                    return {"processed_path": config["data"]["processed_data"]}

            start_time = time.perf_counter()
            logger.info("Started Processing data")

            # Load spaCy model for semantic similarity
            nlp = spacy.load("en_core_web_sm")

            # Read data using polars with optimized settings
            reference_df = pl.read_csv(config["data"]["reference_data"], use_pyarrow=True)
            supplier_df = pl.read_csv(config["data"]["supplier_data"], use_pyarrow=True)

            # Pre-process all room names at once
            reference_df = reference_df.with_columns(
                [pl.col("room_name").map_elements(lambda x: normalize_room_name(x), return_dtype=pl.String).alias("normalized_room_name")]
            )

            supplier_df = supplier_df.with_columns(
                [
                    pl.col("supplier_room_name")
                    .map_elements(lambda x: normalize_room_name(x), return_dtype=pl.String)
                    .alias("normalized_supplier_room_name")
                ]
            )

            # Group and merge data by hotel
            reference_df_grouped = reference_df.group_by("lp_id").agg(
                [pl.struct(["hotel_id", "room_id", "room_name", "normalized_room_name"]).alias("data")]
            )

            supplier_df_grouped = supplier_df.group_by("lp_id").agg(
                [pl.struct(["supplier_room_name", "normalized_supplier_room_name"]).alias("data")]
            )

            # Join only hotels that exist in both datasets
            merged_df = reference_df_grouped.join(supplier_df_grouped, on="lp_id", how="inner")

            # Log hotel statistics - fixed data access pattern
            total_hotels = len(merged_df)
            total_ref_rooms = sum(len(row["data"]) for row in merged_df.iter_rows(named=True))
            total_sup_rooms = sum(len(row["data_right"]) for row in merged_df.iter_rows(named=True))
            logger.info(f"Found {total_hotels} hotels with {total_ref_rooms} reference rooms and {total_sup_rooms} supplier rooms")

            # Prepare all room names for vectorization
            all_room_names: List[str] = []
            for row in merged_df.iter_rows(named=True):
                all_room_names.extend(room["normalized_room_name"] for room in row["data"])
                all_room_names.extend(room["normalized_supplier_room_name"] for room in row["data_right"])

            # Fit TF-IDF vectorizer once with optimized settings
            vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=10000, analyzer="word")
            vectorizer.fit(all_room_names)

            # Save vectorizer
            with open(config["data"]["vectorizer"], "wb") as f:
                pickle.dump(vectorizer, f)

            logger.info(f"Vectorizer fitted in {time.perf_counter() - start_time:.2f} seconds")
            vectorizer_time = time.perf_counter()

            # Check for cached spaCy documents
            spacy_cache_file = os.path.join(config["data"]["data_folder"], "spacy_docs.pkl")
            unique_room_names = set(all_room_names)

            if os.path.exists(spacy_cache_file):
                logger.info("Loading cached spaCy documents...")
                with open(spacy_cache_file, "rb") as f:
                    cached_docs = pickle.load(f)

                # Check if we have all needed documents
                missing_names = unique_room_names - set(cached_docs.keys())
                if missing_names:
                    logger.info(f"Found {len(missing_names)} new room names to process")
                    # Process only missing documents
                    new_docs = {name: nlp(name) for name in missing_names}
                    room_docs = {**cached_docs, **new_docs}
                    # Save updated cache
                    with open(spacy_cache_file, "wb") as f:
                        pickle.dump(room_docs, f)
                else:
                    logger.info("Using cached spaCy documents")
                    room_docs = cached_docs
            else:
                logger.info("No cache found, processing all room names")
                room_docs = {name: nlp(name) for name in unique_room_names}
                # Save cache for future use
                with open(spacy_cache_file, "wb") as f:
                    pickle.dump(room_docs, f)

            logger.info(f"Pre-computed {len(room_docs)} room docs in {time.perf_counter() - vectorizer_time:.2f} seconds")

            matched_pairs, not_matched_pairs = set(), set()

            # Process hotels sequentially
            processed_hotels = 0
            logger.info(f"Processing {total_hotels} hotels sequentially")

            for row in merged_df.iter_rows(named=True):
                processed_hotels += 1
                if processed_hotels % 1000 == 0:
                    logger.info(f"Processed {processed_hotels}/{total_hotels} hotels ({(processed_hotels / total_hotels) * 100:.1f}%)")

                # Process current hotel
                hotel_matches, hotel_not_matches = DataProcessor.process_hotel_rooms(
                    (row["data"], row["data_right"], vectorizer, room_docs, row["lp_id"])
                )
                matched_pairs.update(hotel_matches)
                not_matched_pairs.update(hotel_not_matches)

            logger.info("Combining results from all hotels")

            # Prepare output data
            same_room_name_list = [p for p in matched_pairs if p[0] == p[1]]
            diff_room_name_list = [p for p in matched_pairs if p[0] != p[1]]

            # Sample and combine results
            logger.info(f"Found {len(same_room_name_list)} exact matches and {len(diff_room_name_list)} similar matches")
            matched_cases = diff_room_name_list + same_room_name_list
            not_matched_cases = list(not_matched_pairs)  # Convert set to list

            output = matched_cases + not_matched_cases

            # Save results
            logger.info(f"Total pairs: {len(output)} at {config['data']['processed_data']} in {time.perf_counter() - start_time:.2f} seconds")
            with open(config["data"]["processed_data"], "w") as f:
                json.dump(output, f)

            return {"processed_path": config["data"]["processed_data"]}
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error processing data: {e}")
            raise e
