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
    
    print(f"Vector Latency: {vec_latency:.2f}ms")
    
    # Save to history.json
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
        
    print("Benchmark run logged to benchmarks/history.json successfully!")

if __name__ == "__main__":
    run_benchmarks()
