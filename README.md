# LeafGuard: Lab-to-Field Generalization and Confidence in Tomato Leaf-Disease Classification

Code, settings, and results for a study of how a tomato leaf-disease classifier trained only on laboratory photos (PlantVillage) performs on real-world photos (PlantDoc), whether its confidence ("Trust Score") can flag wrong answers, and whether training-time augmentation narrows the gap.

## Contents

| File | Description |
|---|---|
| `leafguard.ipynb` | Complete executed notebook: data download, cleaning, training, evaluation, figures, demo |
| `requirements.txt` | Exact package versions used for the reported results |
| `results_table.csv` | Pre-audit results for all four methods, using 1,095 lab test images and 743 real-world images |
| `results_chart (2).png`, `reliability (1).png`, `field_confusion (2).png`, `field_confusion_aug (1).png`, `field_examples (2).png` | Figures used in the paper |

## Hardware and runtime

- Tested on Google Colab with a GPU runtime (NVIDIA T4): Runtime → Change runtime type → T4 GPU.
- A full run takes roughly 20–30 minutes on a T4. It also runs on CPU, more slowly.
- No local installation is needed in Colab.

## Software

Versions used (also in `requirements.txt`): torch 2.11.0 (CUDA 13.0 build), torchvision 0.26.0, scikit-learn 1.6.1, numpy 2.1.3, pandas 2.2.3, matplotlib 3.10.0, Pillow 11.3.0, kagglehub 1.0.2. To run outside Colab, install with `pip install -r requirements.txt`, choosing the torch build that matches your hardware.

## Data

The notebook downloads both datasets automatically. They are not stored in this repository.

1. **PlantVillage (lab photos):** Kaggle dataset `emmarex/plantdisease`, downloaded with `kagglehub`. You need a free Kaggle account and an API key file (`kaggle.json`): on kaggle.com, open Settings → API → Create Legacy API Key. The first notebook cell asks you to upload this file.
2. **PlantDoc (real-world photos):** cloned from https://github.com/pratikkayal/PlantDoc-Dataset (CC BY 4.0). Images from both its train and test folders are used for testing only.

## How to reproduce

1. Open `leafguard.ipynb` in Google Colab (File → Upload notebook, or File → Open notebook → GitHub).
2. Select a GPU: Runtime → Change runtime type → T4 GPU.
3. Run everything: Runtime → Run all. Upload `kaggle.json` when prompted.
4. Compare the final audited results printed by the notebook with `results_table_audited.csv` inside `leafguard_artifacts.zip`. These results use 1,095 lab test images and 716 audited real-world images and match the final paper. The repository’s top-level `results_table.csv` contains the earlier, pre-audit results. The ZIP also includes `trust_thresholds_audited.csv` with retained prediction counts and accuracy at each confidence threshold.

## Settings

| Setting | Value |
|---|---|
| Classes | 9 tomato classes present in both datasets (Target Spot dropped: no real-world match) |
| Lab data | Stratified 25% sample of 14,607 PlantVillage images: 2,554 training / 1,095 lab test (70/30 split, seed 42) |
| Real-world test data | 743 PlantDoc tomato images after exact-duplicate removal; 716 after the duplicate audit in Cell 12 (27 removed: 13 with conflicting labels, 14 extra near-duplicate copies); never used for training or tuning |
| Feature extractor | EfficientNet-B0 pretrained on ImageNet (torchvision `EfficientNet_B0_Weights.DEFAULT`), frozen; 1,280 features per image |
| LeafGuard | StandardScaler + LogisticRegression (`max_iter=3000`, `class_weight='balanced'`) |
| LeafGuard + augmentation | Same classifier trained on originals plus 2 augmented copies per training image: RandomResizedCrop(256, scale 0.3–1.0), horizontal flip, rotation ±25°, ColorJitter(0.5, 0.5, 0.5, 0.05), GaussianBlur(kernel 9, sigma 0.5–3.0, p = 0.5); seed 42 |
| Classical baseline | 16-bin HSV color histograms (images resized to 128×128) + RandomForest (`n_estimators=300`, `class_weight='balanced'`, `random_state=42`) |
| Floor | Majority class (most common training class) |
| Metrics | Accuracy; macro-F1; Field Retention Score (real-world macro-F1 ÷ lab macro-F1); Trust Score analysis (expected calibration error with 10 bins, AUROC, and 50/60/70/80/90% thresholds, plus a no-filter baseline) |

## Expected key results (audited test sets: 1,095 lab, 716 real-world)

- LeafGuard: 93.6% accuracy on lab photos; 23.7% accuracy and 0.153 macro-F1 on real-world photos.
- LeafGuard + augmentation: 24.6% accuracy and 0.169 macro-F1 on real-world photos.
- Before the duplicate audit (743 real-world photos): LeafGuard 24.0% / 0.154; LeafGuard + augmentation 24.9% / 0.170.
- Full results: `results_table_audited.csv` and `trust_thresholds_audited.csv` inside `leafguard_artifacts.zip`.

## Saved artifacts (check results without re-running)

`leafguard_artifacts.zip` contains:

| File | Contents |
|---|---|
| `split_lab_train.csv`, `split_lab_test.csv`, `split_real_world.csv` | Every image used, its label, and whether it was kept after the data audit (with the reason if dropped) |
| `predictions_<model>_<split>.csv` | Per-image true label, predicted label, confidence, and probability for each class (LeafGuard, LeafGuard + augmentation, Color + RF; lab test and real-world) |
| `leafguard_classifier.joblib`, `leafguard_aug_classifier.joblib` | Trained classifiers (StandardScaler + LogisticRegression on EfficientNet-B0 features); load with `joblib.load(...)` using scikit-learn 1.6.1 |
| `color_histograms.npz` | Color-histogram features for the Color + RF baseline (refits in seconds with `random_state=42`) |
| `results_table.csv` | Pre-audit results using 1,095 lab test images and 743 real-world images |
| `results_table_audited.csv` | Final audited results using 1,095 lab test images and 716 real-world images; matches the final paper |
| `trust_thresholds_audited.csv` | Audited confidence-threshold results, including retained prediction counts and accuracy |

The data audit (Cell 12) removes real-world images with conflicting labels, keeps one copy of same-label near-duplicates, removes real-world images that nearly match a lab training image, and removes lab test images that nearly match a lab training image.

## Demo

Run the notebook cells in order. The demo is in the cell beginning with `def diagnose(path):`. Change `i` to a number from 0 to 742 and rerun that cell to display the selected image, each model’s prediction, and its Trust Score. The demo selects from the original 743 real-world images; the final evaluation uses the 716 images retained after the audit.

## Credits

PlantVillage (Hughes and Salathé, 2015); PlantDoc (Singh et al., 2020); EfficientNet (Tan and Le, 2019); PyTorch and torchvision; scikit-learn. Claude assisted in debugging code.
