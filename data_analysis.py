import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style untuk plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_and_explore_data():
    """Load dan eksplorasi data"""
    print("=== LOADING AND EXPLORING DATASET ===")
    
    # Load dataset
    df = pd.read_csv('UCI_Credit_Card.csv')
    
    print(f"Shape of dataset: {df.shape}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nDataset Info:")
    print(df.info())
    
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    return df

def visualize_data(df):
    """Visualisasi data untuk EDA"""
    print("\n=== EXPLORATORY DATA ANALYSIS ===")
    
    # Distribusi target variable (pie dan bar)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    target_counts = df['default.payment.next.month'].value_counts()
    plt.pie(target_counts.values, labels=['No Default', 'Default'], autopct='%1.1f%%')
    plt.title('Distribution of Default Payment')
    plt.subplot(1, 2, 2)
    sns.countplot(x='default.payment.next.month', data=df)
    plt.title('Default Payment Count')
    plt.xlabel('Default Payment Next Month')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig('static/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Distribusi umur
    plt.figure(figsize=(6, 4))
    plt.hist(df['AGE'], bins=30, alpha=0.7)
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('static/age_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Distribusi limit balance
    plt.figure(figsize=(6, 4))
    plt.hist(df['LIMIT_BAL'], bins=30, alpha=0.7)
    plt.title('Credit Limit Distribution')
    plt.xlabel('Credit Limit')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('static/limitbal_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Visualisasi hubungan fitur dengan target
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='default.payment.next.month', y='LIMIT_BAL', data=df)
    plt.title('LIMIT_BAL vs Default')
    plt.savefig('static/limitbal_vs_default.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.countplot(x='SEX', hue='default.payment.next.month', data=df)
    plt.title('SEX vs Default')
    plt.savefig('static/sex_vs_default.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.countplot(x='EDUCATION', hue='default.payment.next.month', data=df)
    plt.title('EDUCATION vs Default')
    plt.savefig('static/education_vs_default.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.countplot(x='MARRIAGE', hue='default.payment.next.month', data=df)
    plt.title('MARRIAGE vs Default')
    plt.savefig('static/marriage_vs_default.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('static/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

def preprocess_data(df):
    """Pra-pemrosesan data"""
    print("\n=== DATA PREPROCESSING ===")
    
    # Buat copy dari data
    df_processed = df.copy()
    
    # Handle missing values (jika ada)
    print(f"Missing values before processing: {df_processed.isnull().sum().sum()}")
    if df_processed.isnull().sum().sum() > 0:
        df_processed = df_processed.dropna()
        print("Missing values dropped.")
    else:
        print("No missing values found.")

    # Pastikan tipe data kategorikal sudah numerik
    df_processed['SEX'] = df_processed['SEX'].astype(int)
    df_processed['EDUCATION'] = df_processed['EDUCATION'].astype(int)
    df_processed['MARRIAGE'] = df_processed['MARRIAGE'].astype(int)
    
    print("Categorical variables encoding:")
    print(f"SEX unique values: {df_processed['SEX'].unique()}")
    print(f"EDUCATION unique values: {df_processed['EDUCATION'].unique()}")
    print(f"MARRIAGE unique values: {df_processed['MARRIAGE'].unique()}")
    
    # Check class imbalance
    target_counts = df_processed['default.payment.next.month'].value_counts()
    print(f"\nClass distribution:")
    print(f"Non-default: {target_counts[0]} ({target_counts[0]/len(df_processed)*100:.2f}%)")
    print(f"Default: {target_counts[1]} ({target_counts[1]/len(df_processed)*100:.2f}%)")
    
    return df_processed

def prepare_features(df):
    """Persiapan fitur untuk model"""
    print("\n=== FEATURE PREPARATION ===")
    
    # Pilih fitur yang relevan
    feature_columns = [
        'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE',
        'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    
    X = df[feature_columns]
    y = df['default.payment.next.month']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    return X, y

def train_model(X, y):
    """Training model regresi logistik"""
    print("\n=== MODEL TRAINING ===")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Scaling features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE untuk handle class imbalance
    print("Applying SMOTE to handle class imbalance...")
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    print(f"After SMOTE - Training set size: {X_train_balanced.shape[0]}")
    print(f"After SMOTE - Class distribution: {np.bincount(y_train_balanced)}")
    
    # Train logistic regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_balanced, y_train_balanced)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    return model, scaler, X_test_scaled, y_test, y_pred, y_pred_proba

def evaluate_model(y_test, y_pred, y_pred_proba):
    """Evaluasi model"""
    print("\n=== MODEL EVALUATION ===")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # === Tambahkan visualisasi confusion matrix ===
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Default', 'Default'],
                yticklabels=['No Default', 'Default'])
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.title('Confusion Matrix Hasil Prediksi')
    plt.tight_layout()
    plt.savefig('static/confusion_matrix.png')
    plt.show()
    # === Akhir penambahan ===
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # AUC Score
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nAUC Score: {auc_score:.4f}")
    
    # Business Interpretation
    print("\n=== BUSINESS INTERPRETATION ===")
    tn, fp, fn, tp = cm.ravel()
    print(f"True Negatives (Correctly predicted non-default): {tn}")
    print(f"False Positives (Incorrectly predicted as default): {fp}")
    print(f"False Negatives (Incorrectly predicted as non-default): {fn}")
    print(f"True Positives (Correctly predicted default): {tp}")
    
    print(f"\nFalse Positive Rate: {fp/(fp+tn):.4f}")
    print(f"False Negative Rate: {fn/(fn+tp):.4f}")
    
    print("\nBusiness Impact Analysis:")
    print("- False Positives: Rejecting good customers (lost revenue)")
    print("- False Negatives: Approving bad customers (potential losses)")
    print("- In credit card, False Negatives are typically more costly!")
    
    return cm

def save_model(model, scaler):
    """Simpan model dan scaler"""
    print("\n=== SAVING MODEL ===")
    
    joblib.dump(model, 'credit_card_model.pkl')
    joblib.dump(scaler, 'credit_card_scaler.pkl')
    
    print("Model and scaler saved successfully!")

def main():
    """Main function"""
    print("CREDIT CARD DEFAULT PREDICTION - DATA ANALYSIS")
    print("=" * 50)
    
    # Load and explore data
    df = load_and_explore_data()
    
    # Visualize data
    visualize_data(df)
    
    # Preprocess data
    df_processed = preprocess_data(df)
    
    # Prepare features
    X, y = prepare_features(df_processed)
    
    # Train model
    model, scaler, X_test, y_test, y_pred, y_pred_proba = train_model(X, y)
    
    # Evaluate model
    cm = evaluate_model(y_test, y_pred, y_pred_proba)
    
    # Save model
    save_model(model, scaler)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print("Model training and evaluation completed successfully!")
    print("Model and scaler saved for web application.")

if __name__ == "__main__":
    main()

