"""Simple ML pipeline: classify breast tumors as malignant or benign.

Steps: load data -> train/test split -> scale + logistic regression -> evaluate.
Run with:  python pipeline.py
"""

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42


def main():
    # 1. Load data (ships with scikit-learn, no download needed)
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")

    # 2. Split: 80% train / 20% test, stratified so both keep the class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    # 3. Model: standardize features, then logistic regression.
    #    The scaler lives inside the pipeline so it is fit on training data only.
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

    # 4. 5-fold cross-validation on the training set (the test set stays untouched)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"5-fold CV accuracy: {cv_scores.mean():.1%} (+/- {cv_scores.std():.1%})")

    # 5. Fit on the full training set and evaluate once on the held-out test set
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.1%}")
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=data.target_names))


if __name__ == "__main__":
    main()
