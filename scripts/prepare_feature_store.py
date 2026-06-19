import os

from config import (
    DATA_PATH,
    FEATURE_STORE_PATH
)

from src.inference.feature_store import (
    build_feature_store
)


def validate_data_path():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Update DATA_PATH in config.py or place your dataset at that path."
        )


def main():

    print("\n" + "=" * 70)
    print("PREPARING TRAFFIC FEATURE STORE")
    print("=" * 70)

    validate_data_path()

    build_feature_store(
        data_path=DATA_PATH,
        output_path=FEATURE_STORE_PATH
    )

    if not os.path.exists(FEATURE_STORE_PATH):

        raise FileNotFoundError(
            f"Feature store was not created at: {FEATURE_STORE_PATH}"
        )

    print("\n" + "=" * 70)
    print("FEATURE STORE READY")
    print("=" * 70)

    print("\nGenerated file:")
    print(FEATURE_STORE_PATH)

    print("\nNext command:")
    print("python predict.py")


if __name__ == "__main__":
    main()