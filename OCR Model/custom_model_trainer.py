"""Train the checkpoint-number classifier from a local image dataset."""

import argparse
from pathlib import Path

import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Ziply checkpoint OCR model.")
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing class folders named 1 through 18.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mnist_custom_digits.keras"),
        help="Destination for the trained Keras model.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


def load_dataset(data_dir, img_size=28):
    images = []
    labels = []

    for digit_path in sorted(data_dir.iterdir()):
        if not digit_path.is_dir():
            continue

        label = int(digit_path.name)
        for img_path in digit_path.iterdir():
            image = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv.resize(image, (img_size, img_size))
            image = image.astype("float32") / 255.0
            images.append(np.expand_dims(image, axis=-1))
            labels.append(label - 1)

    if not images:
        raise ValueError(f"No readable training images found under {data_dir}")

    return np.array(images), np.array(labels)


def main():
    args = parse_args()
    images, labels = load_dataset(args.data_dir)
    labels = to_categorical(labels, num_classes=18)

    x_train, x_val, y_train, y_val = train_test_split(
        images,
        labels,
        test_size=0.1,
        random_state=42,
        stratify=labels,
    )

    model = Sequential(
        [
            Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation="relu"),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(18, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=0,
        verbose=1,
        mode="max",
        baseline=1.0,
    )
    model.fit(
        x_train,
        y_train,
        epochs=args.epochs,
        batch_size=16,
        validation_data=(x_val, y_val),
        callbacks=[early_stop],
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)


if __name__ == "__main__":
    main()
