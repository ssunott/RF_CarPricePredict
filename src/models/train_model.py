from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle
import gzip


# Function to train the model and return test data
def train_RFmodel(X, y):
    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Use RandomForest to train model
    model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=None)
    model = model.fit(X_train, y_train)

    # Save the trained model
    with gzip.open('models/RFmodel.pkl', 'wb', compresslevel=6) as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    return model, X_test, y_test
