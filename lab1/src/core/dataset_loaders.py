
def load_raw_text_data( dataset_path: str):
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {dataset_path}")
    except Exception as e:
        raise RuntimeError(f"Error reading file {dataset_path}: {e}")
