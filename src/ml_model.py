from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import random
import pickle
import joblib
import pandas as pd


def train_random_forest():
    filepath = "../datasets/business_category_train_1.csv"
    df = pd.read_csv(filepath)
    indices_of_interest = df["Category"].value_counts(
    )[df["Category"].value_counts() >= 10].index
    df = df[df["Category"].isin(indices_of_interest)]
    le = preprocessing.LabelEncoder()

    le.fit(["UNKNOWN"] + list(df["Category"]))
    df["Category (encoded)"] = le.transform(df["Category"])
    random.seed(123)

    raw_train = df
    desc_vectorizer = CountVectorizer(analyzer="word", max_features=100)
    training_bag_of_words = desc_vectorizer.fit_transform(raw_train["Expense"])

    feature_names = desc_vectorizer.get_feature_names_out()
    x_train = pd.DataFrame(training_bag_of_words.toarray(),
                           columns=feature_names).astype(int)

    feature_names = desc_vectorizer.get_feature_names_out()

    rf = RandomForestClassifier()
    rf.fit(x_train, raw_train["Category (encoded)"])

    # Save the fitted vectorizer
    joblib.dump(desc_vectorizer, 'vectorizer.pkl')
    joblib.dump(rf, 'random_forest_model.pkl')

def predict(summary: pd.DataFrame) -> pd.DataFrame:
    rf = joblib.load('random_forest_model.pkl')

    desc_vectorizer = joblib.load('vectorizer.pkl')

    test_bag_of_words = desc_vectorizer.transform(
        summary["Item Description"])
    feature_names = desc_vectorizer.get_feature_names_out()
    x_summary = pd.DataFrame(
        test_bag_of_words.toarray(), columns=feature_names).astype(int)
    
    # Make predictions using the loaded model
    summary["category"] = rf.predict(x_summary)
    
    return summary

if __name__ == "__main__":
    # train and save the model
    train_random_forest()
