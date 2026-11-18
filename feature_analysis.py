
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pipeline = joblib.load("retainiq_model.joblib")

preprocessor = pipeline.named_steps["preprocessor"]
model = pipeline.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()

if hasattr(model, "coef_"):
    values = model.coef_[0]
    analysis = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": values,
        "Absolute_Importance": np.abs(values),
        "Direction": np.where(
            values > 0, "Higher churn tendency",
            "Lower churn tendency"
        )
    })
    title = "Logistic Regression: Top Influential Features"
else:
    values = model.feature_importances_
    analysis = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": values,
        "Absolute_Importance": values,
        "Direction": "Feature importance"
    })
    title = "Random Forest: Top Influential Features"

analysis = analysis.sort_values(
    "Absolute_Importance", ascending=False
)

analysis.to_csv("feature_importance.csv", index=False)

top = analysis.head(15).sort_values("Coefficient")

plt.figure(figsize=(10, 7))
plt.barh(top["Feature"], top["Coefficient"])
plt.title(title)
plt.xlabel("Coefficient / Importance")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()

print("Feature analysis saved!")
print(analysis.head(15).to_string(index=False))
