#!/usr/bin/env bash
# fetch_demo_images.sh - download eight APTOS photographs for the prototype demo.
#
# All eight come from the TEST split of splits.csv, so the model never saw them during training or
# when the final model was chosen. They cover every stage and three different cameras, so the demo
# shows a healthy eye, borderline cases and advanced disease.
#
# It needs the Kaggle API token at ~/.kaggle/kaggle.json and the APTOS 2019 competition rules
# accepted by the same account (the account that ran the notebooks already has both).
#
# Run it from the project folder with the virtual environment active:
#     source venv/bin/activate
#     bash app/fetch_demo_images.sh
#
# Computer Vision CW1 - Diabetic Retinopathy Stage Detection - Mohamed Shafran
set -euo pipefail
cd "$(dirname "$0")/demo_images" 2>/dev/null || { mkdir -p "$(dirname "$0")/demo_images"; cd "$(dirname "$0")/demo_images"; }

command -v kaggle >/dev/null 2>&1 || python3 -m pip install --quiet kaggle

# id_code and the stage the dataset gives it; the app shows the stage next to its own answer
IMAGES="901a3552fe26 0
8e6df9eedcd8 0
b6a0e348a01e 1
70ed3ec68b94 2
78a577c3e0bf 2
b2ffa3e18559 3
6f4719c6bb4b 4
a182b5b191de 4"

echo "id_code,diagnosis" > labels.csv
while read -r id stage; do
    if [ ! -f "$id.png" ]; then
        echo "downloading $id.png (stage $stage)"
        kaggle competitions download -c aptos2019-blindness-detection -f "train_images/$id.png" --force
        if [ -f "$id.png.zip" ]; then                  # Kaggle zips some single-file downloads
            unzip -o -q "$id.png.zip" && rm -f "$id.png.zip"
        fi
    fi
    echo "$id,$stage" >> labels.csv
done <<< "$IMAGES"

echo
ls -1 ./*.png 2>/dev/null | wc -l | xargs echo "photographs in app/demo_images:"
echo "If nothing downloaded, open https://www.kaggle.com/competitions/aptos2019-blindness-detection/data"
echo "once, accept the rules, and run this script again."
