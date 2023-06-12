import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    st.title("MACHILEARN.com")

    # Untuk Upload Data
    file = st.file_uploader("Upload CSV file", type=["csv"], key="file_uploader")

    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df.head())

        st.subheader("Train a Classifier")
        target_col = st.selectbox("Select the target column", df.columns)
        features = df.drop(target_col, axis=1)
        target = df[target_col]

        # Untuk Non Numerik Value
        if not features.applymap(lambda x: isinstance(x, (int, float))).all().all():
            st.write(
                "Warning: Non-numeric values detected in the features. Removing non-numeric columns."
            )
            features = features.select_dtypes(include=["int", "float"])

        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        st.write("Accuracy: ", accuracy)

        # User I
        st.subheader("Make Predictions")
        input_data = {}
        for col in features.columns:
            value = st.number_input(f"Enter {col}", key=col)
            input_data[col] = value

        input_df = pd.DataFrame([input_data])

        # Display Predictions
        if st.button("Predict"):
            prediction = model.predict(input_df)
            st.write("Prediction: ", prediction[0])


if __name__ == "__main__":
    main()
