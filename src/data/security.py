"""
Security Module
Encrypt and decrypt sensitive data at rest using Fernet (symmetric encryption).
Guideline: All sensitive data must be encrypted at rest and in transit.
"""

import os
import logging
from cryptography.fernet import Fernet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_key(key_path: str = "configs/.encryption_key") -> bytes:
    """Generate and save a new encryption key."""
    key = Fernet.generate_key()
    os.makedirs(os.path.dirname(key_path), exist_ok=True)
    with open(key_path, "wb") as f:
        f.write(key)
    logger.info(f"Encryption key generated and saved to {key_path}")
    return key


def load_key(key_path: str = "configs/.encryption_key") -> bytes:
    """Load encryption key from file."""
    if not os.path.exists(key_path):
        logger.warning("No encryption key found. Generating a new one.")
        return generate_key(key_path)
    with open(key_path, "rb") as f:
        return f.read()


def encrypt_file(
    input_path: str, output_path: str, key_path: str = "configs/.encryption_key"
):
    """Encrypt a file at rest."""
    key = load_key(key_path)
    fernet = Fernet(key)

    with open(input_path, "rb") as f:
        data = f.read()

    encrypted = fernet.encrypt(data)

    with open(output_path, "wb") as f:
        f.write(encrypted)

    logger.info(f"Encrypted {input_path} -> {output_path}")


def decrypt_file(
    input_path: str, output_path: str, key_path: str = "configs/.encryption_key"
):
    """Decrypt an encrypted file."""
    key = load_key(key_path)
    fernet = Fernet(key)

    with open(input_path, "rb") as f:
        encrypted = f.read()

    decrypted = fernet.decrypt(encrypted)

    with open(output_path, "wb") as f:
        f.write(decrypted)

    logger.info(f"Decrypted {input_path} -> {output_path}")


if __name__ == "__main__":
    # Example: encrypt the raw dataset
    generate_key()
    encrypt_file("data/raw/creditcard.csv", "data/raw/creditcard.csv.enc")
    print("Data encrypted at rest.")
