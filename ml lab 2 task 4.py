import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.feature_selection import chi2
survey_df = pd.read_csv("Survey.csv")  

print(survey_df.describe(include="all"))
 
sns.countplot(x="Q1", data=survey_df)
plt.show()


survey_df.fillna(survey_df.mode().iloc[0], inplace=True)


class_counts = survey_df["TARGET"].value_counts()
minority = survey_df[survey_df["TARGET"] == class_counts.idxmin()]
majority = survey_df[survey_df["TARGET"] == class_counts.idxmax()]
majority_downsampled = resample(majority, replace=False, n_samples=len(minority), random_state=42)
balanced_survey = pd.concat([minority, majority_downsampled])

categorical_features = pd.get_dummies(balanced_survey.drop("TARGET", axis=1))
chi_scores, p_vals = chi2(categorical_features, balanced_survey["TARGET"])
chi_series = pd.Series(chi_scores, index=categorical_features.columns).sort_values(ascending=False)
print("Top Chi-Square Features:\n", chi_series.head())

dummy_encoded = pd.get_dummies(survey_df, drop_first=True)
print("Dummy Encoded Shape:", dummy_encoded.shape)

survey_df = survey_df.drop_duplicates()                        
survey_df = survey_df.drop_duplicates(subset=["Q1", "Q2"])  