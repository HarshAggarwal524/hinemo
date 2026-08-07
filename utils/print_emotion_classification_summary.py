import json

BASE_PATH = "/Users/harshaggarwal/Projects_4/hinemo_project/models"
models = [
    "bert-base-multilingual-cased",
    "muril-base-cased",
    "xlm-roberta-base"
]
states   = ["pretrained", "finetuned"]
EMOTIONS = ["anger", "disgust", "joy", "sadness"]

model_short = {
    "bert-base-multilingual-cased": "mBERT",
    "muril-base-cased"            : "MuRIL",
    "xlm-roberta-base"            : "XLM-R"
}

print(f"\n{'='*110}")
print("EMOTION CLASSIFICATION PROBING — 4k")
print(f"{'='*110}")
print(f"{'Model+State':<25} {'Global Peak':<13} {'Peak F1':<10} "
      f"{'Anger':<7} {'Disgust':<9} {'Joy':<7} {'Sadness':<9} "
      f"{'Low-λ Peak':<12} {'High-λ Peak'}")
print("-" * 105)

for model in models:
    for state in states:
        path = f"{BASE_PATH}/phase7b_probing_results_emotion_classification_4k/{model}/emotion_classification_probe_{state}.json"
        with open(path) as f:
            r = json.load(f)

        label         = f"{model_short[model]} {state}"
        global_peak   = r["global_peak_layer"]
        global_f1     = r["global_f1_per_layer"][global_peak - 1]
        emotion_peaks = [str(r["peak_layer_per_emotion"][e]) for e in EMOTIONS]
        low_peak      = r["lambda_split"]["low_lambda"]["peak_layer"]
        high_peak     = r["lambda_split"]["high_lambda"]["peak_layer"]

        print(f"{label:<25} {global_peak:<13} {global_f1:<10.4f} "
              f"{emotion_peaks[0]:<7} {emotion_peaks[1]:<9} "
              f"{emotion_peaks[2]:<7} {emotion_peaks[3]:<9} "
              f"{low_peak:<12} {high_peak}")