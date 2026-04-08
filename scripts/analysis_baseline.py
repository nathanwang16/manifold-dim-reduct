
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def check_baseline():
    print("Loading Phase 2 data...")
    try:
        # Load embeddings and labels
        # Assuming these are aligned
        embeddings = np.load('phase2_results/pca_embeddings.npy')
        labels = np.load('phase2_results/labels.npy')
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Labels shape: {labels.shape}")
        
        # Flatten labels if needed
        if len(labels.shape) > 1:
            labels = labels.ravel()
            
        # Subsample if too large for quick check
        if len(labels) > 50000:
            indices = np.random.choice(len(labels), 50000, replace=False)
            embeddings = embeddings[indices]
            labels = labels[indices]
            print(f"Subsampled to {len(labels)} samples")
            
        # Split
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
        
        # 1. Linear Probe (Logistic Regression)
        print("\nRunning Linear Probe (Logistic Regression)...")
        lr = LogisticRegression(max_iter=1000, C=1.0)
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        acc_lr = accuracy_score(y_test, y_pred_lr)
        print(f"Linear Probe Accuracy: {acc_lr:.4f} ({acc_lr*100:.2f}%)")
        
        # 2. k-NN (Non-linear local geometry)
        print("\nRunning k-NN (k=30)...")
        knn = KNeighborsClassifier(n_neighbors=30)
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        acc_knn = accuracy_score(y_test, y_pred_knn)
        print(f"k-NN Accuracy: {acc_knn:.4f} ({acc_knn*100:.2f}%)")
        
        return acc_lr, acc_knn
        
    except Exception as e:
        print(f"Error: {e}")
        return 0, 0

if __name__ == "__main__":
    check_baseline()






