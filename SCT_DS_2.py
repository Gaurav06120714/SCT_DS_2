import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv("titanic/train.csv")

print("Shape:", train.shape)
print("\nMissing values:\n", train.isnull().sum())
print("\nBasic stats:\n", train.describe())

train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna(train["Embarked"].mode()[0])
train.drop(columns=["Cabin"], inplace=True)

survival_rate = train.groupby("Pclass")["Survived"].mean() * 100
gender_survival = train.groupby("Sex")["Survived"].mean() * 100

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Titanic Dataset — Exploratory Data Analysis", fontsize=16, fontweight="bold", y=1.01)

axes[0, 0].pie(
    train["Survived"].value_counts(),
    labels=["Did Not Survive", "Survived"],
    colors=["#E57373", "#81C784"],
    autopct="%1.1f%%",
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5}
)
axes[0, 0].set_title("Overall Survival Rate")

train["Age"].hist(bins=30, ax=axes[0, 1], color="#42A5F5", edgecolor="white")
axes[0, 1].set_title("Age Distribution")
axes[0, 1].set_xlabel("Age")
axes[0, 1].set_ylabel("Count")

survival_rate.plot(kind="bar", ax=axes[0, 2], color=["#EF9A9A", "#FFCC80", "#80CBC4"], edgecolor="white")
axes[0, 2].set_title("Survival Rate by Passenger Class")
axes[0, 2].set_xlabel("Pclass")
axes[0, 2].set_ylabel("Survival Rate (%)")
axes[0, 2].set_xticklabels(["1st", "2nd", "3rd"], rotation=0)

gender_survival.plot(kind="bar", ax=axes[1, 0], color=["#F48FB1", "#90CAF9"], edgecolor="white")
axes[1, 0].set_title("Survival Rate by Gender")
axes[1, 0].set_xlabel("Sex")
axes[1, 0].set_ylabel("Survival Rate (%)")
axes[1, 0].set_xticklabels(["Female", "Male"], rotation=0)
axes[1, 0].set_ylim(0, 100)

train[train["Survived"] == 1]["Age"].hist(bins=25, ax=axes[1, 1], alpha=0.7, label="Survived", color="#66BB6A")
train[train["Survived"] == 0]["Age"].hist(bins=25, ax=axes[1, 1], alpha=0.7, label="Not Survived", color="#EF5350")
axes[1, 1].set_title("Age vs Survival")
axes[1, 1].set_xlabel("Age")
axes[1, 1].legend()

sns.heatmap(
    train[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr(),
    annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1, 2],
    linewidths=0.5, linecolor="white"
)
axes[1, 2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.savefig("task02_titanic_eda.png", dpi=150, bbox_inches="tight")
plt.show()
print("Chart saved as task02_titanic_eda.png")
