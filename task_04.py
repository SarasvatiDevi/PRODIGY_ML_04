import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Define gesture labels (fixed spacing)
gesture_names = [
    'palm', 'L', 'fist', 'fist moved', 'thumb',
    'index', 'OK', 'palm moved', 'C', 'down'
]

def load_images(data_dir, img_size=(64, 64)):
    """
    Loads grayscale images and their labels from the dataset folder.
    """
    X, y = [],[]
    person_folders = sorted(os.listdir(data_dir))

    for subject_folder in person_folders:
        subject_path = os.path.join(data_dir, subject_folder)
        if not os.path.isdir(subject_path):
            continue

        for gesture_folder in sorted(os.listdir(subject_path)):
            gesture_path = os.path.join(subject_path, gesture_folder)
            if not os.path.isdir(gesture_path):
                continue

            try:
                gesture_label = int(gesture_folder[:2]) - 1  # Fixed index to match gesture_names
                if gesture_label not in range(10):
                    continue
            except ValueError:
                print(f"Skipping {gesture_folder}: cannot parse label.")
                continue

            for img_name in sorted(os.listdir(gesture_path)):
                if img_name.endswith(".png"):
                    img_path = os.path.join(gesture_path, img_name)
                    try:
                        img = Image.open(img_path).resize(img_size).convert('L')
                        X.append(np.array(img))
                        y.append(gesture_label)
                    except Exception as e:
                        print(f"Skipping {img_path}: {e}")

    print(f"Loaded {len(X)} images.")
    return np.array(X), np.array(y)

# Load and preprocess data
data_dir = "C:/Users/Sanjina/Downloads/leapGestRecog"
x, y = load_images(data_dir)
print("Unique labels:", np.unique(y))  # Helpful debug print

x = x / 255.0
x = x.reshape(-1, 64, 64, 1)
y = to_categorical(y, num_classes=10)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(10, activation="softmax")
])

# Compile model
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"]
)

model.summary()

# Train model
history = model.fit(
    x_train, y_train,
    validation_data=(x_test, y_test),
    epochs=15,
    batch_size=64,
    verbose=2
)

# Evaluate model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Visualize prediction
def visualize_prediction(index):
    prediction = model.predict(x_test[index].reshape(1, 64, 64, 1))
    predicted_class = np.argmax(prediction)
    true_class = np.argmax(y_test[index])
    confidence = np.max(prediction)

    plt.imshow(x_test[index].reshape(64, 64), cmap="gray")
    plt.title(
        f"Predicted: {gesture_names[predicted_class]} ({confidence:.2f})\nTrue: {gesture_names[true_class]}"
    )
    plt.axis("off")
    plt.show()

# Show one prediction
visualize_prediction(5)

# Save model
model.save("hand_gesture_model.h5")
print("Model saved as 'hand_gesture_model.h5'")

# Plot training results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("training_results.png")
plt.show()
print("Training plot saved as 'training_results.png'")
