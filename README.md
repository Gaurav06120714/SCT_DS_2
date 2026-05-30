# SCT_DS_2 — Titanic EDA (Exploratory Data Analysis)

## Task Overview
Perform data cleaning and exploratory data analysis on the Titanic dataset. Explore relationships between variables and identify patterns and trends.

## Dataset
**Source:** Kaggle — Titanic: Machine Learning from Disaster  
**Files:** `titanic/train.csv`, `titanic/test.csv`, `titanic/gender_submission.csv`  
**Records:** 891 passengers (training set)

## What the Script Does
- Loads the Titanic training dataset and inspects shape, missing values, and stats
- **Data Cleaning:**
  - Fills missing `Age` values with the median age
  - Fills missing `Embarked` with the mode
  - Drops the `Cabin` column (77%+ missing values)
  - Engineers `FamilySize` feature (`SibSp + Parch + 1`)
- **Visualizations — Panel 1 (6-panel figure):**
  1. Overall survival rate (pie chart)
  2. Age distribution (histogram)
  3. Survival rate by passenger class (bar chart)
  4. Survival rate by gender (bar chart)
  5. Age vs Survival comparison (overlapping histograms)
  6. Correlation heatmap (includes FamilySize)
- **Visualizations — Panel 2 (extended analysis):**
  7. Fare distribution (histogram)
  8. Fare outliers (boxplot)
  9. Survival rate by port of embarkation
- **Visualizations — Panel 3:**
  10. Survival rate by family size

## Key Findings
- Women survived at a ~74% rate vs ~19% for men
- 1st class passengers survived at ~63% vs ~24% for 3rd class
- Passengers from Cherbourg (C) had the highest survival rate
- Fare is heavily right-skewed with significant outliers (max ~$512)
- Small families (2–4) survived better than solo travelers or large families
- FamilySize and Fare show the strongest positive correlation with survival

## Project Structure
```
SCT_DS_2/
├── titanic/
│   ├── train.csv                          ← main training dataset
│   ├── test.csv
│   └── gender_submission.csv
├── images/
│   ├── task02_titanic_eda.png             ← 6-panel EDA output
│   ├── task02_titanic_extended.png        ← fare + embarkation analysis
│   └── task02_titanic_family.png          ← family size analysis
├── SCT_DS_2.py                            ← main script
├── Titanic_Cleaned.csv                    ← cleaned dataset export
└── README.md
```

## Libraries Used
| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, feature engineering |
| `matplotlib` | Multi-panel plotting |
| `seaborn` | Heatmap, boxplot |

## How to Run
```bash
cd SCT_DS_2
python SCT_DS_2.py
```

## Output
Three figures saved as PNG files showing survival patterns across gender, class, age, fare, embarkation port, and family size.

---
**Internship:** SkillCraft Technology — Data Science  
**Task:** 02 | Data Cleaning & EDA
