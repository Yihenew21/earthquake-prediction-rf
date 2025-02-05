import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path="data/earthquake_data.csv"):
    """
    Load and preprocess earthquake dataset.
    """
    # Load dataset
    df = pd.read_csv(file_path)

    # Select relevant features
    df = df[['mag', 'depth', 'latitude', 'longitude', 'nst']]  
    df.dropna(inplace=True)  # Remove missing values

    # Define features and target
    X = df[['depth', 'latitude', 'longitude', 'nst']]
    y = df['mag']  # Target: Earthquake magnitude

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler

