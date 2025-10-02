# TabPFN for Student Performance Prediction

## 📋 Project Overview

This project reproduces the TabPFN paper and extends it to the Education domain for student performance prediction. TabPFN is a novel approach for tabular classification that uses transformer-based models trained on synthetic data for zero-shot learning.

**Original Paper:** TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second
- **Authors:** Noah Hollmann, Samuel Müller, Katharina Eggensperger, Frank Hutter
- **Published:** NeurIPS 2022
- **Paper URL:** https://arxiv.org/abs/2207.01848
- **Original Code:** https://github.com/automl/TabPFN

**Group Details:**
- **Batch:** TYBTech ML
- **Group Number:** 01
- **Members:**
  1. Aryan Shinde - 16014223018
 

---

## 🎯 Problem Statement

**Challenge:**
Traditional machine learning models for tabular data require extensive hyperparameter tuning and substantial training data.

**TabPFN Solution:**
Uses transformers pre-trained on synthetic tabular data to enable zero-shot learning without hyperparameter tuning.

**Our Extension:**
Apply TabPFN to predict student academic performance (Pass/Fail) in the Education domain and compare with traditional baseline models.

---

## 📊 Datasets Used

### Original Datasets (Reproduction Phase)
Successfully reproduced paper results on 4 benchmark datasets:

1. **Credit-G** (OpenML ID: 31)
   - Samples: 1000 | Features: 20 | Classes: 2
   - Accuracy: 0.7800

2. **Diabetes** (OpenML ID: 37)
   - Samples: 768 | Features: 8 | Classes: 2
   - Accuracy: 0.7474

3. **Vehicle** (OpenML ID: 54)
   - Samples: 846 | Features: 18 | Classes: 4
   - Accuracy: 0.8676

4. **Breast-W** (OpenML ID: 15)
   - Samples: 699 | Features: 9 | Classes: 2
   - Accuracy: 0.9543

### New Dataset (Extension Phase)
- **Dataset Name:** Student Performance Dataset (Portuguese Students)
- **Source:** UCI Machine Learning Repository / Kaggle
- **Dataset URL:** https://archive.ics.uci.edu/ml/datasets/Student+Performance
- **Domain:** Education
- **Task:** Binary classification (Pass/Fail prediction)
- **Original Target:** G3 (final grade, 0-20)
- **Converted Target:** Pass (G3 ≥ 10) / Fail (G3 < 10)
- **Samples:** 649 students
- **Original Features:** 33
- **Final Features:** 29 (after removing G1, G2, G3)
- **Classes:** 2

**Features Include:**
- Demographics: age, sex, address, family size
- Social: parent education, family relationships, free time, social activities
- School: study time, failures, extra support, absences

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip or conda

### Install Dependencies
```bash
