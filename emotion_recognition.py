import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------------------------------------------------
# 1. Dataset Configuration & Label Mappings
# ---------------------------------------------------------

# Standard RAVDESS emotion code mapping
RAVDESS_EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def extract_mfcc(file_path: str, n_mfcc: int = 40, max_pad_len: int = 174) -> np.ndarray:
    """
    Loads an audio file, extracts MFCCs, and pads/truncates to a fixed length.
    """
    # Load audio sampled at 22050 Hz
    audio, sr = librosa.load(file_path, sr=22050, duration=3.0, offset=0.5)

    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    # Pad or truncate along the time dimension to ensure constant matrix shape
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :max_pad_len]

    return mfcc.T  # Shape: (time_steps, n_mfcc) -> (174, 40)


def generate_synthetic_samples(num_samples: int = 120):
    """
    Generates dummy audio matrices for quick verification if no audio folder exists.
    """
    print("[INFO] No dataset directory found or empty. Generating synthetic MFCC samples for testing...")
    emotions = ["happy", "sad", "angry", "neutral"]
    X = []
    y = []

    for _ in range(num_samples):
        # Create random feature matrix with shape (174, 40)
        X.append(np.random.normal(0, 1, (174, 40)))
        y.append(np.random.choice(emotions))

    return np.array(X), np.array(y)


def load_dataset(data_dir: str):
    """
    Traverses the directory, parses labels safely, and extracts MFCC features.
    Supports RAVDESS filename format (e.g., 03-01-05-01-01-01-01.wav).
    """
    if not os.path.exists(data_dir):
        return generate_synthetic_samples()

    audio_files = glob.glob(os.path.join(data_dir, "**", "*.wav"), recursive=True)

    if len(audio_files) == 0:
        return generate_synthetic_samples()

    print(f"[INFO] Found {len(audio_files)} .wav audio files in '{data_dir}'. Extracting MFCCs...")

    X, y = [], []
    for file_path in audio_files:
        filename = os.path.basename(file_path)

        # Parse RAVDESS naming convention
        parts = filename.split(".")[0].split("-")
        if len(parts) >= 3 and parts[2] in RAVDESS_EMOTIONS:
            emotion = RAVDESS_EMOTIONS[parts[2]]
        else:
            # Fallback label extraction from parent folder name
            emotion = os.path.basename(os.path.dirname(file_path))

        try:
            features = extract_mfcc(file_path)
            X.append(features)
            y.append(emotion)
        except Exception as e:
            print(f"[WARNING] Could not parse {file_path}: {e}")
            continue

    if len(X) == 0:
        return generate_synthetic_samples()

    return np.array(X), np.array(y)


# ---------------------------------------------------------
# 2. Deep Learning Model (CNN + LSTM Architecture)
# ---------------------------------------------------------

def build_cnn_lstm_model(input_shape, num_classes):
    """
    Combines 1D CNN layers for local spatial/spectral feature extraction
    and LSTM layers for temporal sequence modeling.
    """
    model = models.Sequential([
        # 1D Convolutional Blocks for local frequency pattern extraction
        layers.Conv1D(64, kernel_size=5, strides=1, padding="same", activation="relu", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        layers.Conv1D(128, kernel_size=5, strides=1, padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.3),

        # LSTM Layer for temporal speech patterns
        layers.LSTM(64, return_sequences=False),
        layers.Dropout(0.3),

        # Dense Classifier Head
        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


# ---------------------------------------------------------
# 3. Training & Evaluation Pipeline
# ---------------------------------------------------------

def train(data_dir: str, epochs: int = 15, batch_size: int = 16):
    print("[1/4] Loading audio data and extracting MFCCs...")
    X, y_text = load_dataset(data_dir)

    print(f"[INFO] Dataset shape: {X.shape}, Labels count: {len(y_text)}")

    # Encode textual emotion labels to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_text)
    num_classes = len(label_encoder.classes_)

    # Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[2/4] Building CNN-LSTM model for {num_classes} emotion classes: {list(label_encoder.classes_)}...")
    input_shape = (X_train.shape[1], X_train.shape[2])  # (174, 40)
    model = build_cnn_lstm_model(input_shape, num_classes)
    model.summary()

    print("[3/4] Training Model...")
    model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    print("[4/4] Model Evaluation...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Loss: {loss:.4f}")
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%\n")

    # Classification Metrics
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))


if __name__ == "__main__":
    # Point this to your dataset folder (e.g., folder with Actor_01, Actor_02, etc.)
    # If folder doesn't exist, it will safely run synthetic data to demonstrate execution.
    DATASET_DIRECTORY = "dataset"
    train(data_dir=DATASET_DIRECTORY, epochs=15, batch_size=16)