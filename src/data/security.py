"""
Security Module
Encrypt and decrypt sensitive data at rest using Fernet (symmetric encryption).
Guideline: All sensitive data must be encrypted at rest and in transit.

The default key path comes from ``security.encryption_key_path`` in
configs/config.yaml; callers can still pass an explicit path to override.
"""

import os
import logging

import yaml
from cryptography.fernet import Fernet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_CONFIG_PATH = "configs/config.yaml"
_FALLBACK_KEY_PATH = "configs/.encryption_key"


def _default_key_path() -> str:
    """Read ``security.encryption_key_path`` from config, with a fallback."""
    if not os.path.exists(_CONFIG_PATH):
        return _FALLBACK_KEY_PATH
    with open(_CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg.get("security", {}).get("encryption_key_path", _FALLBACK_KEY_PATH)


def generate_key(key_path: str = None) -> bytes:
    """Generate and save a new encryption key."""
    key_path = key_path or _default_key_path()
    key = Fernet.generate_key()
    parent = os.path.dirname(key_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(key_path, "wb") as f:
        f.write(key)
    logger.info(f"Encryption key generated and saved to {key_path}")
    return key


def load_key(key_path: str = None) -> bytes:
    """Load encryption key from file."""
    key_path = key_path or _default_key_path()
    if not os.path.exists(key_path):
        logger.warning("No encryption key found. Generating a new one.")
        return generate_key(key_path)
    with open(key_path, "rb") as f:
        return f.read()


def encrypt_file(input_path: str, output_path: str, key_path: str = None):
    """Encrypt a file at rest."""
    key = load_key(key_path)
    fernet = Fernet(key)

    with open(input_path, "rb") as f:
        data = f.read()

    encrypted = fernet.encrypt(data)

    with open(output_path, "wb") as f:
        f.write(encrypted)

    logger.info(f"Encrypted {input_path} -> {output_path}")


def decrypt_file(input_path: str, output_path: str, key_path: str = None):
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
    # Example: encrypt the raw dataset (path comes from config.data.raw_path).
    if os.path.exists(_CONFIG_PATH):
        with open(_CONFIG_PATH, "r") as _f:
            _cfg = yaml.safe_load(_f) or {}
        _raw = _cfg.get("data", {}).get("raw_path", "data/raw/creditcard.csv")
    else:
        _raw = "data/raw/creditcard.csv"
    generate_key()
    encrypt_file(_raw, _raw + ".enc")
    print("Data encrypted at rest.")
