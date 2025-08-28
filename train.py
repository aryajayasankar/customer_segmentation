import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import mlflow
import mlflow.sklearn

def train_model(n_clusters, random_state=42):
    """
    Trains a K-Means clustering model and logs the experiment with MLflow.
    """
    # Load the data
    df = pd.read_csv("Mall_Customers.csv")

    # Select features for segmentation
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Start an MLflow run
    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_param("n_clusters", n_clusters)
        mlflow.log_param("random_state", random_state)

        # Train the model
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        model.fit(X_scaled)

        # Calculate a performance metric (e.g., Silhouette Score)
        score = silhouette_score(X_scaled, model.labels_)

        # Log metrics
        mlflow.log_metric("silhouette_score", score)

        # Log the trained model as an artifact
        mlflow.sklearn.log_model(model, "kmeans_model")
        print(f"Logged run with n_clusters={n_clusters} and silhouette score={score}")


if __name__ == "__main__":
    # Example usage: train with different numbers of clusters
    train_model(n_clusters=3)
    train_model(n_clusters=5)
    train_model(n_clusters=7)