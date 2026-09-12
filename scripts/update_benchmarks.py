import json
import os
import re
import time

def run_benchmarks():
    print("Running automated benchmark suite...")
    start_time = time.time()
    
    # Measure vector search speed
    from sauti_rag import SautiEngine
    engine = SautiEngine()
    
    vec_start = time.time()
    results = engine.search("noma", top_k=1)
    vec_latency = (time.time() - vec_start) * 1000
    total_latency = (time.time() - start_time) * 1000
    
    # Update benchmarks/history.json
    os.makedirs("benchmarks", exist_ok=True)
    history_file = "benchmarks/history.json"
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []
            
    history.append({
        "timestamp": time.time(),
        "vector_latency_ms": round(vec_latency, 2),
        "total_latency_ms": round(total_latency, 2)
    })
    
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)
        
    print(f"Logged run #{len(history)} to benchmarks/history.json")

    # Update README.md visualizations
    readme_path = "README.md"
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            content = f.read()

        # Update historical run count in README
        updated_content = re.sub(
            r"Historical Tracking:\*\* Latency logs saved to `benchmarks/history.json` \(\d+ total run\(s\) recorded\)",
            f"Historical Tracking:** Latency logs saved to `benchmarks/history.json` ({len(history)} total run(s) recorded)",
            content
        )

        with open(readme_path, "w") as f:
            f.write(updated_content)
        print("Updated README.md benchmark metrics!")

if __name__ == "__main__":
    run_benchmarks()
