import requests
import os
import sys

if __name__ == "__main__":
    # https://huggingface.co/deepghs/imgutils-models/resolve/main/person_detect/person_detect_plus_v1.1_best_m.pt?download=true
    url = "https://huggingface.co/deepghs/imgutils-models/resolve/main/person_detect/person_detect_plus_v1.1_best_m.pt?download=true"
    if os.path.exists("person_detect_plus_v1.1_best_m.pt"):
        print("Model already exists")
    else:
        print("Downloading model...")
        head = requests.head(url)
        r = requests.get(url, allow_redirects=True)
        with open("person_detect_plus_v1.1_best_m.pt", "wb") as f:
            f.write(r.content)
        print("Model downloaded")
