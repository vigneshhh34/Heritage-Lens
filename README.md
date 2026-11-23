# Heritage Lens

**Architectural Style Recognition and AI-Assisted Repair of Historical Building Photographs**

Course project for **Introduction to Data Science (IDSE CS3006)**, Department of Computer Science and Engineering, IIITDM Kancheepuram.
Guide: **Prof. Jagadeesh Kakarla** | Team: **Group-26**

[![Open the classifier in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/vigneshhh34/Heritage-Lens/blob/main/notebooks/heritage_lens_classifier.ipynb)

---

## What this project does

Old photographs of buildings are often faded, scratched or partly destroyed. Heritage Lens addresses two practical questions about a single photograph:

1. **Which architectural style does it show?** A ResNet-50 classifier recognises 43 styles (for example Gothic, Mughal, Dravidian, Khmer, Art Deco, Baroque).
2. **Can a damaged area be repaired plausibly?** The user paints over the damaged part, optionally types what was there, and Stable Diffusion inpainting fills only that area with four candidate images.

A CLIP-based filter rejects uploads that do not show a building (for example faces or food), so the classifier does not give meaningless answers.

> **Important:** repaired regions are AI-generated illustrations, not reconstructions of the true original. The application labels them as such.

## Scope

The project started from 3D reconstruction of buildings from sparse historical data (see the literature review in the report). After discussing it with the course instructor, 3D rendering was not required and the generation of restored pictures was encouraged, so the work concentrates on single photographs. Acoustic data, which appeared in the original working title, was dropped.

## Main results

All numbers come from the experiments in the notebooks and the report.

### Style classifier (test set, 1,465 images, 43 styles)

| Metric | Value |
|---|---|
| Top-1 accuracy | 0.799 |
| Top-3 accuracy | 0.925 |
| Macro F1 | 0.806 |
| Weighted F1 | 0.798 |

The easiest styles are visually distinctive (Maya, Balinese, Khmer, Minangkabau, Edo Period, Joseon Dynasty, Hausa, Ancient Egyptian). The hardest are overlapping Western styles (Beaux-Arts, Colonial, Edwardian, Brutalist, Bauhaus, Postmodern, International style, Georgian).

### Repair experiments (synthetic damage, 12 photographs, original known)

| Finding | Evidence |
|---|---|
| Redrawing a damaged image from its edge map (ControlNet) is worse than leaving it untouched | Worse on PSNR, SSIM and LPIPS in every damage type |
| Mask-based Stable Diffusion inpainting works | Error inside the damaged region 58.9 to 27.5 (0-255 scale); LPIPS 0.297 to 0.094 |
| Soft blending and colour matching | No gain |
| Automatic BLIP captions as prompts | No reliable gain (better in 6 of 12 photos) |
| Naming the missing building in the prompt | Better fill in 10 of 12 photos |
| Old photos: scratch removal + denoising + CLAHE | Modest gain (PSNR 17.3 dB against 15.6 dB, greyscale comparison) |
| Generative sharpening of old photos | Invents architectural detail; lower SSIM |

The experiments are small (12 photographs, one run each) and use synthetic damage, so differences smaller than the seed-to-seed variation (LPIPS about 0.007) should not be over-interpreted. See the report for the full analysis and limitations.

## How it works

```
Input photo
  -> Building filter (CLIP ViT-B/32, accept if score >= 0.2)
  -> Style classifier (ResNet-50, 43 classes) -> top-3 styles + short description
  -> Optional repair: painted mask + text -> Stable Diffusion inpainting (4 seeds)
                       -> unmasked pixels are kept from the original
  -> Optional old-photo cleanup: scratch removal -> denoising -> CLAHE
```

## Repository contents

```
Heritage-Lens/
├── README.md
├── notebooks/
│   ├── heritage_lens_classifier.ipynb     # data exploration, training, evaluation of the classifier
│   ├── 02_restoration_and_app.ipynb       # clean notebook: reload models, verify results, launch the app
│   └── 03_original_experiments.ipynb      # building filter, repair experiments (Rounds 1-5), app development
├── results/                                # CSV result tables and example figures from the experiments
└── report/
    └── Heritage_Lens_Report.pdf            # full project report
```

## How to run

## Download the trained model and results

Trained classifier (`best.pt`, `classes.json`), result tables and the report are available here:
[Google Drive folder]([PASTE-YOUR-LINK-HERE](https://drive.google.com/drive/folders/1DeX--Mjz75my_Hcs9Zj2EvmTSV0o2Ilb?usp=drive_link))

To run the app, copy `best.pt` and `classes.json` into `MyDrive/heritage-lens/models/` in your own Google Drive, then run `notebooks/02_restoration_and_app.ipynb`.
The notebooks were developed in **Google Colab with a T4 GPU**.

1. Open a notebook in Colab (use the badge above for the classifier) and select **Runtime > Change runtime type > T4 GPU**.
2. **Classifier:** run `heritage_lens_classifier.ipynb` from top to bottom. It downloads the dataset from Kaggle (see below), trains the ResNet-50 for 8 epochs and saves the model and class list to Google Drive.
3. **Repair and app:** run `02_restoration_and_app.ipynb`. It needs the trained model files (`best.pt`, `classes.json`) in `MyDrive/heritage-lens/models/`, and it loads Stable Diffusion inpainting and CLIP from Hugging Face. The first run downloads several GB of model files. The last cell launches the Gradio app.

Model weights and the dataset are **not** stored in this repository because of their size. Free Colab GPU access is limited and not guaranteed, so a GPU may be unavailable at times; the repair part is too slow on a CPU to be practical.

The notebooks use fixed random seeds (data split 42, generation seeds 0 to 3). Kaggle credentials must be stored in Colab Secrets and must never be written into a notebook.

## Data and credits

- **Style dataset:** *International Architectural Styles - Combined* by jungseolin on Kaggle (CC BY-SA 4.0), which merges earlier datasets by Danci and by Xu et al. (ECCV 2014) with Wikimedia Commons images. 45 styles, 14,823 images; two classes with fewer than 150 images (Pueblo, Jain) were removed, leaving 43 styles and 14,648 images (train 11,718 / validation 1,465 / test 1,465).
  https://www.kaggle.com/datasets/jungseolin/international-architectural-styles-combined
- **Photographs for the repair experiments:** 12 photographs from Wikimedia Commons with open licences. File names, authors and licences are listed in Appendix B of the report.
- **Qualitative test photographs:** seven photographs collected by the authors from public web sources, used only for illustration. Their sources were not recorded.
- **Models and libraries:** ResNet-50 (He et al.), CLIP (Radford et al.), Stable Diffusion (Rombach et al.), ControlNet (Zhang et al.), BLIP (Li et al.), LPIPS (Zhang et al.), PyTorch, Hugging Face diffusers, OpenCV, scikit-learn, Gradio. Full references are in the report.

Images in the notebooks and figures remain subject to the licences and credits of their original sources.

## Limitations

- The classifier knows only 43 styles and always answers with one of them; its confidence is not calibrated.
- Some dataset labels are noisy, and near-duplicate images across the splits were not checked for.
- Repair results are plausible illustrations, not recovered originals, and were evaluated on synthetic damage with a small sample.
- The building filter was tested on a small set of non-building photographs.
- No 3D output; this is proposed as future work for cases with several views of the same building.

## Team (Group-26)

| Name | Roll number |
|---|---|
| G Sai Vigneshwar Reddy | CS24B1082 |
| A Vikhilesh Krishna | CS24B1077 |
| N Ramya Satwika | CS24I1011 |
| P Rahul | CS24B1009 |

Guide: Prof. Jagadeesh Kakarla

## Citation of this project

> Group-26, *Heritage Lens: Architectural Style Recognition and AI-Assisted Repair of Historical Building Photographs*, IDSE CS3006 project report, IIITDM Kancheepuram, November 2026.
