import re
import os


def normalize_room_name(name: str) -> str:
    # original_name = name
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", "", name)  # remove punctuation/special chars
    name = re.sub(r"\s+", " ", name).strip()  # collapse multiple spaces
    # print(f"Normalized {original_name} -> {name}")
    return name


def check_in_ignore_words(word):
    if word in [
        "access",
        "accessible",
        "air",
        "allowed",
        "and",
        "basic",
        "bath",
        "bathroom",
        "bathrooms",
        "bed",
        "bedroom",
        "bedrooms",
        "beds",
        "cabin",
        "chalet",
        "city",
        "classic",
        "comfort",
        "companion",
        "complimentary",
        "dinner",
        "first",
        "floor",
        "garden",
        "guest",
        "halfboard",
        "hearing",
        "high",
        "hot",
        "included",
        "is",
        "j",
        "jacuzzi",
        "japanese",
        "junior",
        "kitchen",
        "luxury",
        "mandatory",
        "menaggio",
        "microwave",
        "mobility",
        "mountain",
        "multiple",
        "netflixdis",
        "no",
        "non",
        "nonsmoking",
        "not",
        "ocean",
        "one",
        "open",
        "or",
        "panoramic",
        "partial",
        "pet",
        "pets",
        "pool",
        "resort",
        "romantic",
        "room",
        "royal",
        "s",
        "sea",
        "shared",
        "signature",
        "smoking",
        "sofa",
        "spa",
        "standard",
        "suite",
        "superior",
        "supreme",
        "terrace",
        "third",
        "three",
        "tub",
        "twin",
        "two",
        "view",
        "village",
        "w",
        "waterfront",
        "western",
        "with",
    ] + [str(i) for i in range(0, 10)]:
        return True
    return False


def get_next_experiment_folder(base_path: str) -> str:
    existing_folders = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d)) and d.startswith("exp")]

    if not existing_folders:
        next_num = 1
    else:
        # Extract numbers from folder names and find max
        nums = [int(folder[3:]) for folder in existing_folders]
        next_num = max(nums) + 1

    os.makedirs(os.path.join(base_path, f"exp{next_num}"), exist_ok=True)
    return os.path.join(base_path, f"exp{next_num}")
