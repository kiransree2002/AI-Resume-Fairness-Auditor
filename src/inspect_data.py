import pandas as pd
import os

# Define file paths
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RESUME_PATH = os.path.join(DATA_DIR, "Resume.csv")
POSTINGS_PATH = os.path.join(DATA_DIR, "postings.csv")

def inspect_resumes():
    print("=" * 60)
    print("📄 1. INSPECTING RESUME DATASET")
    print("=" * 60)
    
    if not os.path.exists(RESUME_PATH):
        print(f"❌ Error: Could not find {RESUME_PATH}")
        return
    
    df_resumes = pd.read_csv(RESUME_PATH)
    print(f"✅ Total Resumes Loaded: {len(df_resumes)}")
    print(f"📋 Columns: {list(df_resumes.columns)}")
    print("\n🔍 Missing Values Count:")
    print(df_resumes.isnull().sum())
    
    print("\n📊 Unique Categories Breakdown (Top 10):")
    print(df_resumes['Category'].value_counts().head(10))
    
    print("\n🔎 Sample Resume Text Preview (First 300 characters):")
    first_text = df_resumes['Resume_str'].iloc[0]
    print(f'"{first_text[:300]}..."')


def inspect_postings():
    print("\n" + "=" * 60)
    print("💼 2. INSPECTING LINKEDIN JOB POSTINGS DATASET")
    print("=" * 60)
    
    if not os.path.exists(POSTINGS_PATH):
        print(f"❌ Error: Could not find {POSTINGS_PATH}")
        return
    
    # Read only first 500 rows to keep it fast
    df_postings = pd.read_csv(POSTINGS_PATH, nrows=500)
    print(f"✅ Sample Postings Loaded: {len(df_postings)}")
    print(f"📋 Columns: {list(df_postings.columns)}")
    
    # Find columns related to title and description
    title_col = [c for c in df_postings.columns if 'title' in c.lower()]
    desc_col = [c for c in df_postings.columns if 'desc' in c.lower() or 'text' in c.lower()]
    
    print(f"\n📌 Detected Job Title Column(s): {title_col}")
    print(f"📌 Detected Description Column(s): {desc_col}")


if __name__ == "__main__":
    inspect_resumes()
    inspect_postings()