import json
from itertools import combinations

BASE_PATH = "/Users/harshaggarwal/Projects_4/hinemo_project/models"
models = [
    "bert-base-multilingual-cased",
    "muril-base-cased",
    "xlm-roberta-base"
]
states   = ["pretrained", "finetuned"]
EMOTIONS = ["anger", "disgust", "joy", "sadness"]

def print_global_table(sample_size, folder):
    print(f"\n{'='*70}")
    print(f"GLOBAL λ PROBING — {sample_size}")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'State':<12} {'LSL':<6} {'Peak R²'}")
    print("-" * 65)
    for model in models:
        for state in states:
            path = f"{BASE_PATH}/{folder}/{model}/probing_{state}.json"
            with open(path) as f:
                r = json.load(f)
            print(f"{model:<35} {state:<12} {r['LSL']:<6} {r['peak_r2']:.4f}")

def print_emotion_table(sample_size, folder):
    print(f"\n{'='*90}")
    print(f"EMOTION-CONDITIONED λ PROBING — {sample_size}")
    print(f"{'='*90}")
    print(f"{'Model':<35} {'State':<12} {'ELDS':<8} {'Anger':<6} {'Disgust':<8} {'Joy':<6} {'Sadness'}")
    print("-" * 85)
    for model in models:
        for state in states:
            path = f"{BASE_PATH}/{folder}/{model}/emotion_probing_{state}.json"
            with open(path) as f:
                r = json.load(f)
            lsl = r["lsl_per_emotion"]
            print(f"{model:<35} {state:<12} {r['ELDS']:<8.4f} "
                  f"{lsl['anger']:<6} {lsl['disgust']:<8} "
                  f"{lsl['joy']:<6} {lsl['sadness']}")

print_global_table("3k", "phase7a_probing_results_3k_lambda")
print_emotion_table("3k", "phase7a_probing_results_3k_lambda")
print_global_table("4k", "phase7a_probing_results_4k_lambda")
print_emotion_table("4k", "phase7a_probing_results_4k_lambda")