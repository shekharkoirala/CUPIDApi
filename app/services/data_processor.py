from typing import Dict, Any
from app.utils.config_loader import ConfigLoader
from app.utils.logger import CustomLogger
import polars as pl
import traceback
from app.utils.helper import normalize_room_name, check_in_ignore_words
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
import json
import time
import pickle
import os


class DataProcessor:
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

            reference_df = pl.read_csv(config["data"]["reference_data"])
            supplier_df = pl.read_csv(config["data"]["supplier_data"])

            reference_df_grouped = reference_df.group_by("lp_id").agg(pl.struct(["hotel_id", "room_id", "room_name"]).alias("data"))
            supplier_df_grouped = supplier_df.group_by("lp_id").agg(pl.struct(["supplier_room_name"]).alias("data"))
            merged_df = reference_df_grouped.join(supplier_df_grouped, on="lp_id", how="inner")

            # import pdb; pdb.set_trace()

            vectorizer = TfidfVectorizer(stop_words="english")
            all_room_names = [normalize_room_name(room["room_name"]) for hotel in merged_df["data"] for room in hotel] + [
                normalize_room_name(room["supplier_room_name"]) for hotel in merged_df["data_right"] for room in hotel
            ]
            vectorizer.fit(all_room_names)
            with open(config["data"]["vectorizer"], "wb") as f:
                pickle.dump(vectorizer, f)
            logger.info(f"Vectorizer fitted in {time.perf_counter() - start_time:.2f} seconds and saved to {config['data']['vectorizer']}")

            matched_pairs, not_matched_pairs = set(), set()
            for row in merged_df.iter_rows(named=True):
                ref_names = [normalize_room_name(r["room_name"]) for r in row["data"]]
                sup_names = [normalize_room_name(s["supplier_room_name"]) for s in row["data_right"]]

                ref_vecs = vectorizer.transform(ref_names)
                sup_vecs = vectorizer.transform(sup_names)

                sims = cosine_similarity(sup_vecs, ref_vecs)

                for i, s_name in enumerate(sup_names):
                    j = sims[i].argmax()
                    score = sims[i, j]
                    r_name = ref_names[j]

                    if score >= 0.5:
                        unmatched_words = [w for w in r_name.split() if w not in s_name.split()]
                        # checking reference words in supplier, suppliers tends to have more words
                        ignore_filter = [w for w in unmatched_words if not check_in_ignore_words(w)]
                        # only care if unmatched words are in ignore list eg: no pets, no smoking, etc
                        if not ignore_filter:  # that means all unmatched words are in ignore list
                            highest_match = {"s": s_name, "r": r_name}
                            if await DataProcessor.analyze_strings(highest_match, unmatched_words):
                                matched_pairs.add((s_name, r_name, 1))
                                continue

                    not_matched_pairs.add((s_name, r_name, 0))

            same_room_name_list = [p for p in matched_pairs if p[0] == p[1]]  # same room name: all of cases in dataset
            diff_room_name_list = [p for p in matched_pairs if p[0] != p[1]]  # not so much

            matched_cases = random.sample(same_room_name_list, 2000 - len(diff_room_name_list)) + diff_room_name_list
            not_matched_cases = random.sample(list(not_matched_pairs), 2000)

            output = matched_cases + not_matched_cases
            random.shuffle(output)

            logger.info(f"Total pairs: {len(output)} at {config['data']['processed_data']} in {time.perf_counter() - start_time:.2f} seconds")
            with open(config["data"]["processed_data"], "w") as f:
                json.dump(output, f)

            # Your processing logic here
            return {"processed_path": config["data"]["processed_data"]}
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error processing data: {e}")
            raise e

    @staticmethod
    async def analyze_strings(data, word_to_check):
        if not word_to_check:
            return True
        s = data["s"].lower()
        r = data["r"].lower()

        # Check if any word from the list is at the start or end of 'r'

        combined = " ".join(word_to_check)
        starts_with_word = r.startswith(combined)
        ends_with_word = r.endswith(combined)

        s_words = s.split()
        r_words = r.split()

        if starts_with_word:
            w_index = r_words.index(word_to_check[-1])
            check_index = w_index + 1
            if check_index >= len(r_words):
                print(data, word_to_check)
            if r_words[check_index] != s_words[0]:
                return False

        if ends_with_word:
            w_index = r_words.index(word_to_check[0])
            check_index = w_index - 1
            if r_words[check_index] != s_words[-1]:
                return False

        # If no replacements found, return True if the word is added at the start or end
        return starts_with_word or ends_with_word
