# FILE: run.py
# Purpose: Command-line entry to run the full multi-agent pipeline.

from crew.crew import run_pipeline

if __name__ == "__main__":
    print("🧠 Starting AI Newsroom pipeline…")
    results = run_pipeline()
    print("\n✅ Done!")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("\nOpen the files under the 'artifacts/' folder to view your results.")
