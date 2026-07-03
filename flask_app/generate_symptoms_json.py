"""
generate_symptoms_json.py
─────────────────────────
Run once to produce  flask_app/api/symptoms.json  from symptom_columns.csv.

Usage:
    python generate_symptoms_json.py                    # default paths
    python generate_symptoms_json.py --csv path/to/symptom_columns.csv
    python generate_symptoms_json.py --dataset path/to/dataset_Disease.csv  (re-extracts from raw CSV)

Output format:
    {
      "meta": { total, generated_at, source },
      "categories": [
        {
          "id": "neurological",
          "label": "Neurological & Mental Health",
          "symptoms": [
            { "id": "anxiety_and_nervousness", "label": "Anxiety and Nervousness", "index": 0 },
            ...
          ]
        },
        ...
      ],
      "flat": [          <-- ordered list matching model feature vector exactly
        { "id": "anxiety_and_nervousness", "label": "Anxiety and Nervousness", "category": "neurological", "index": 0 },
        ...
      ],
      "id_to_index": { "anxiety_and_nervousness": 0, ... }
    }
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone

# ─── Category rules (checked in order; first match wins) ───────────────────
# Each rule: (category_id, category_label, list_of_keyword_fragments)
CATEGORY_RULES = [
    ("skin",          "Skin & Hair",              [
        "skin","rash","itch","lesion","blister","hive","pigment","sweat",
        "hair","nail","bump","boil","bruise","pallor","jaundice","acne",
        "pimple","wart","flushing","complexion","red_skin","yellow_skin",
        "dry_or_flaky","dry_skin","scaling","ulcer_on_skin","skin_lesion",
        "skin_rash","skin_irritation","skin_peeling","blackhead",
    ]),
    ("eye",           "Eye & Vision",             [
        "eye","vision","blind","pupil","lid","lacrimation",
        "white_discharge_from_eye","cross-eyed",
    ]),
    ("ear",           "Ear & Hearing",            [
        "ear","hearing","tinnitus","pus_draining_from_ear",
        "diminished_hearing","ringing_in_ear",
    ]),
    ("throat_mouth",  "Throat, Mouth & Nose",     [
        "throat","voice","swallow","speak","nasal","coryza",
        "mouth","lip","tongue","dry_lips","mouth_ulcer","mouth_dryness",
        "sneezing","nose","congestion","sore_throat",
    ]),
    ("respiratory",   "Respiratory",              [
        "breath","cough","wheez","sputum","apnea","hemoptysis",
        "breathing_fast","shortness_of_breath","pus_in_sputum",
        "chest_tightness","chest_pain","sharp_chest_pain",
        "smoking","flulike","flu",
    ]),
    ("cardiovascular","Cardiovascular",           [
        "heart","palpitation","heartbeat","cardiac","chest_pounding",
        "irregular_heartbeat","rapid_heart",
    ]),
    ("neurological",  "Neurological",             [
        "dizz","vertigo","faint","confu","memory","unconscious",
        "seizure","nerve","brain","tremor","spasm","involuntary",
        "movement","migrain","headache","paresthesia","loss_of_sensation",
        "slurring","numbness","tingling","head_","abnormal_","stiffness_all",
    ]),
    ("mental_health", "Mental Health & Behaviour",[
        "anxiety","depress","mental","emotional","stress","hallucin",
        "psychotic","behavior","mood","insomnia","sleep","suicid",
        "restless","anger","temper","low_self","abusing_alcohol","drug_abuse",
        "irritable","hostile","excessive_anger","emotional_symptoms",
        "depressive","irritab",
    ]),
    ("digestive",     "Digestive & Abdominal",    [
        "stomach","abdomen","abdominal","nausea","vomit","diarr","constip",
        "bowel","bloat","gas","flatulence","digest","appetite","stool",
        "liver","bile","rectal","anus","melena","regurgitation",
        "blood_in_stool","sharp_abdominal","infant_spitting","difficulty_eating",
    ]),
    ("urinary",       "Urinary & Kidney",         [
        "urine","urinary","bladder","kidney","urethr","void",
        "frequent_urination","excessive_urination","retention_of_urine",
    ]),
    ("musculoskeletal","Muscles & Joints",        [
        "joint","knee","hip","elbow","shoulder","wrist","ankle","arthrit",
        "muscle","weak","fatigue","tired","cramp","stiffness","arm_stiff",
        "leg_stiff","back_weak","elbow_weak","neck_stiff","arm_swelling",
        "hand_or_finger","foot_or_toe","leg_pain","back_pain","arm_pain",
        "swollen_lymph",
    ]),
    ("pain_general",  "Pain & Discomfort",        [
        "pain","ache","tender","burning","sharp_pain","sore ",
    ]),
    ("reproductive",  "Reproductive & Sexual",    [
        "testic","scrotal","scrotum","vagina","menstrual","menstruation",
        "penis","prostate","impotence","infertility","uterine","pelvic",
        "breast","pregnancy","vulvar","hot_flash","reproductive","groin",
        "swelling_of_scrotum","mass_in_scrotum","symptoms_of_the_scrotum",
        "absence_of_menstruation","frequent_menstruation","recent_pregnancy",
    ]),
    ("pediatric",     "Paediatric & Growth",      [
        "infant","child","growth","pediatric","irritable_infant",
        "infant_feeding","symptoms_of_infant","lack_of_growth","excessive_growth",
        "feet_turned","irregular_belly",
    ]),
    ("systemic",      "Systemic & General",       [
        "fever","chill","temperature","feeling_ill","weight","swelling",
        "edema","lymphedema","fluid_retention","allergic","mass","lump",
        "swollen","neck_mass","jaw_swelling","neck_swelling","leg_swelling",
        "foot_or_toe_lump","back_mass","hand_or_finger_lump","peripheral_edema",
        "feeling_cold","feeling_hot","recent_weight","underweight","weight_gain",
        "feeling_hot_and_cold","flulike_syndrome",
    ]),
]


def categorise(symptom_id: str) -> str:
    """Return category_id for a symptom slug using keyword matching."""
    s = symptom_id.lower()
    for cat_id, _label, keywords in CATEGORY_RULES:
        if any(kw in s for kw in keywords):
            return cat_id
    return "other"


def to_label(symptom_id: str) -> str:
    """Convert snake_case id to Title Case human label."""
    return symptom_id.replace("_", " ").title()


def load_from_csv(csv_path: str) -> list[str]:
    """Load symptom column names from symptom_columns.csv (one per line, header='symptom')."""
    symptoms = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        for row in reader:
            if row and row[0].strip():
                symptoms.append(row[0].strip())
    return symptoms


def load_from_dataset(dataset_path: str) -> list[str]:
    """Extract symptom columns directly from the raw dataset CSV (slower, ~190 MB)."""
    with open(dataset_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
    headers = [
        h.strip().lower().replace(" ", "_") for h in headers
    ]
    # Drop the target column
    target_aliases = {"disease", "diseases", "prognosis", "label", "target"}
    symptoms = [h for h in headers if h not in target_aliases]
    return symptoms


def build_json(symptoms: list[str], source: str) -> dict:
    # Build category buckets while preserving original order within each bucket
    cat_map: dict[str, list] = {}
    for cat_id, _, _ in CATEGORY_RULES:
        cat_map[cat_id] = []
    cat_map["other"] = []

    for idx, sym in enumerate(symptoms):
        cat_id = categorise(sym)
        cat_map[cat_id].append({
            "id":    sym,
            "label": to_label(sym),
            "index": idx,         # position in model feature vector — critical for inference
        })

    # Build categories list (skip empty buckets)
    cat_label_map = {cid: lbl for cid, lbl, _ in CATEGORY_RULES}
    cat_label_map["other"] = "Other / Miscellaneous"

    categories = []
    for cat_id, cat_label, _ in CATEGORY_RULES:
        if cat_map[cat_id]:
            categories.append({
                "id":       cat_id,
                "label":    cat_label,
                "count":    len(cat_map[cat_id]),
                "symptoms": cat_map[cat_id],
            })
    if cat_map["other"]:
        categories.append({
            "id":       "other",
            "label":    "Other / Miscellaneous",
            "count":    len(cat_map["other"]),
            "symptoms": cat_map["other"],
        })

    # Flat list — exactly mirrors model feature vector order
    flat = []
    for cat in categories:
        for sym in cat["symptoms"]:
            flat.append({
                "id":       sym["id"],
                "label":    sym["label"],
                "category": cat["id"],
                "index":    sym["index"],
            })
    flat.sort(key=lambda x: x["index"])   # restore original column order

    # id → index lookup (for fast inference encoding)
    id_to_index = {sym["id"]: sym["index"] for sym in flat}

    return {
        "meta": {
            "total":        len(symptoms),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source":       source,
            "note":         (
                "The 'index' field is the feature vector position used during model training. "
                "Always use it when constructing the binary symptom array for inference."
            ),
        },
        "categories":  categories,
        "flat":        flat,
        "id_to_index": id_to_index,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate symptoms.json for the Flask API")
    parser.add_argument("--csv",     default="symptom_columns.csv",
                        help="Path to symptom_columns.csv (default)")
    parser.add_argument("--dataset", default=None,
                        help="Re-extract from raw dataset CSV instead of symptom_columns.csv")
    parser.add_argument("--out",     default="api/symptoms.json",
                        help="Output path (default: api/symptoms.json)")
    args = parser.parse_args()

    # Load symptoms
    if args.dataset:
        if not os.path.exists(args.dataset):
            sys.exit(f"[ERROR] Dataset not found: {args.dataset}")
        print(f"[INFO] Extracting from dataset: {args.dataset}")
        symptoms = load_from_dataset(args.dataset)
        source = os.path.basename(args.dataset)
    else:
        if not os.path.exists(args.csv):
            sys.exit(f"[ERROR] CSV not found: {args.csv}  (use --csv or --dataset)")
        print(f"[INFO] Loading from: {args.csv}")
        symptoms = load_from_csv(args.csv)
        source = os.path.basename(args.csv)

    print(f"[INFO] {len(symptoms)} symptoms found")

    # Build and write JSON
    data = build_json(symptoms, source)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\n[OK] Written → {args.out}")
    print(f"     Total symptoms : {data['meta']['total']}")
    print(f"     Categories     : {len(data['categories'])}")
    for cat in data["categories"]:
        print(f"       {cat['id']:20s}  {cat['count']:3d} symptoms")


if __name__ == "__main__":
    main()