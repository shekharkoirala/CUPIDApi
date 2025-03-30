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


class DataProcessor:
    @staticmethod
    async def process_data(data: bool) -> Dict[str, Any]:
        logger = CustomLogger.get_logger()
        try:
            config = ConfigLoader.get_config()
            reference_df = pl.read_csv(config["data"]["reference_data"])
            supplier_df = pl.read_csv(config["data"]["supplier_data"])

            reference_df_struct = reference_df.with_columns(
                [pl.struct(["hotel_id", "room_id", "room_name"]).alias("data")]
            )
            reference_df_grouped = reference_df_struct.group_by("lp_id").agg(
                pl.col("data")
            )
            supplier_df_struct = supplier_df.with_columns(
                [
                    pl.struct(
                        [
                            "core_room_id",
                            "core_hotel_id",
                            "supplier_room_id",
                            "supplier_room_name",
                        ]
                    ).alias("data")
                ]
            )
            supplier_df_grouped = supplier_df_struct.group_by("lp_id").agg(
                pl.col("data")
            )
            merged_df = reference_df_grouped.join(
                supplier_df_grouped, on="lp_id", how="inner"
            )

            cleaned_data = []
            for row in merged_df.iter_rows(named=True):
                reference = []
                supplier = []
                for r in row["data"]:
                    reference.append(normalize_room_name(r["room_name"]))
                for s in row["data_right"]:
                    supplier.append(normalize_room_name(s["supplier_room_name"]))
                cleaned_data.append({"reference": reference, "supplier": supplier})

            all_room_names = [
                name
                for hid in range(len(cleaned_data))
                for name in cleaned_data[hid]["supplier"]
                + cleaned_data[hid]["reference"]
            ]  # hid = hotel_id
            vectorizer = TfidfVectorizer(stop_words="english").fit(all_room_names)

            matched_pairs = set()
            not_matched_pairs = set()
            for hotels in cleaned_data:
                supplier = hotels["supplier"]
                reference = hotels["reference"]
                # supplier = ["apartment for 5 people"]
                # reference = ["apartment for 4 people"]
                sims = cosine_similarity(
                    vectorizer.transform(supplier), vectorizer.transform(reference)
                )
                for i, sup_name in enumerate(supplier):
                    highest_score = 0
                    highest_match = {}
                    for j, ref_name in enumerate(reference):
                        if sims[i, j] >= highest_score:
                            highest_match = {"s": sup_name, "r": ref_name}
                            highest_score = sims[i, j]

                    if highest_score >= 0.5:  # and highest_score<0.9999999999:
                        r_words = highest_match["r"].split()
                        no_match = [
                            w for w in r_words if w not in highest_match["s"].split()
                        ]
                        no_match_count = len(no_match)
                        for w in no_match:
                            if check_in_ignore_words(w):
                                no_match_count -= 1
                        # print(r_words, no_match)
                        if not no_match_count:
                            # print(f"{highest_score:.2f}", highest_match, no_match)
                            if await DataProcessor.analyze_strings(
                                highest_match, no_match
                            ):
                                # print(f"{highest_score:.2f}", highest_match["s"], "\t", highest_match["r"], no_match ,)
                                matched_pairs.add(
                                    (highest_match["s"], highest_match["r"], 1)
                                )
                                continue

                    not_matched_pairs.add((highest_match["s"], highest_match["r"], 0))

            # check if good pairs are same names
            same_room_name_list = []
            diff_room_name_list = []
            for pair in list(matched_pairs):
                supplier_name, reference_name, _label = pair
                if supplier_name != reference_name:
                    diff_room_name_list.append(pair)
                else:
                    same_room_name_list.append(pair)

            matched_cases = (
                random.sample(same_room_name_list, 2000 - len(diff_room_name_list))
                + diff_room_name_list
            )
            not_matched_cases = random.sample(list(not_matched_pairs), 2000)
            output = matched_cases + not_matched_cases
            random.shuffle(output)

            logger.info(
                f"Total pairs: {len(output)} at {config['data']['processed_data']}"
            )
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
