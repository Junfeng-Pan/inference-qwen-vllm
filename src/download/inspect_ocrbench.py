from modelscope.msdatasets import MsDataset

def inspect_dataset():
    # Load a small subset or streaming to inspect structure
    print("Loading OCRBench dataset...")
    try:
        # subset_name might be needed, but let's try default
        ds = MsDataset.load('evalscope/OCRBench', split='test') # OCRBench usually only has test split
        
        print(f"Dataset loaded. Size: {len(ds)}")
        print("First sample keys:", ds[0].keys())
        print("First sample example:", ds[0])
        
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    inspect_dataset()
