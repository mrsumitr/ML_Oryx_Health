# ML_Oryx_Health  
Document Intake Service for Receipt Understanding

## Overview
This project implements a document intake pipeline that converts **OCR outputs from receipt images** into **structured JSON** suitable for downstream workflows such as billing, analytics, and auditing.

The solution is designed using the **CORD-v2 dataset** and focuses on **system design, data processing, and engineering decisions**, rather than achieving state-of-the-art model performance.

---

## Problem Statement
- **Input:** OCR outputs (text tokens with bounding boxes)
- **Output:** Structured JSON representing receipt information

The primary objective is to design a **data processing function** that maps OCR outputs into structured JSON.

---

## Dataset
- **CORD-v2 (Receipt Dataset)**  
  https://huggingface.co/datasets/naver-clova-ix/cord-v2

The dataset provides:
- Receipt images
- OCR tokens and bounding boxes
- Ground-truth structured annotations

> OCR is assumed to be performed upstream. This project does not implement OCR.

---

## Approach
1. Load OCR tokens and bounding boxes from CORD-v2  
2. Convert document-level annotations into token-level BIO labels  
3. Train a **LayoutLMv3** token classification model  
4. Use model predictions to enable structured JSON generation through post-processing

---

## Project Structure

ML_Oryx_Health/
├── Design.md          # Assumptions, evaluation, limitations
├── inference.py       # Inference on OCR tokens (no OCR step)
├── README.md
├── requirements.txt   # Project dependencies
└── train.py           # Training script (LayoutLMv3)



---

## Environment Setup

### Create and activate a conda environment
```bash
conda create -n oryx-health python=3.10
conda activate oryx-health
```
## Install dependencies
pip install -r requirements.txt

---

## Training

Run the training script:

python train.py

--- 

## Inference

After training, run:
python inference.py

---

## Evaluation

Token-level F1 score and accuracy using seqeval

Qualitative inspection of extracted receipt fields

Comparison with CORD-v2 ground-truth annotations
