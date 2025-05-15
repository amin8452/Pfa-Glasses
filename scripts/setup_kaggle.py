import os
import json
import argparse

def setup_kaggle_credentials(username, key):
    """
    Set up Kaggle API credentials.
    
    Args:
        username (str): Kaggle username.
        key (str): Kaggle API key.
    """
    # Create Kaggle directory if it doesn't exist
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    
    # Create kaggle.json file
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    with open(kaggle_json, "w") as f:
        json.dump({"username": username, "key": key}, f)
    
    # Set permissions
    os.chmod(kaggle_json, 0o600)
    
    print(f"Kaggle API credentials saved to {kaggle_json}")
    print("You can now use the Kaggle API to download datasets.")

def main():
    parser = argparse.ArgumentParser(description="Set up Kaggle API credentials")
    parser.add_argument("--username", type=str, required=True, help="Kaggle username")
    parser.add_argument("--key", type=str, required=True, help="Kaggle API key")
    args = parser.parse_args()
    
    setup_kaggle_credentials(args.username, args.key)

if __name__ == "__main__":
    main()
