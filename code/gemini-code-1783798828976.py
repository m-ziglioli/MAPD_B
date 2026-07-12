import time
import dask.bag as db
from dask.distributed import Client, LocalCluster
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler
import numpy as np

# Import your unmodified classifier and helper function from "dist comp times 0.py"
from kmeans_parallel import kmeans_parallel

def calculate_inertia(X_bag, centroids):
    centroids_arr = np.vstack(centroids)
    return X_bag.map(lambda x: np.min(np.linalg.norm(x - centroids_arr, axis=1)**2)).sum().compute()

def main():
    print("Loading dataset...")
    dataset = fetch_kddcup99(as_frame=True, subset='SA', percent10=True) 
    df_numeric = dataset.data.select_dtypes(include=[np.number]).dropna().astype(float)
    X_numpy = StandardScaler().fit_transform(df_numeric.values)
    
    # ---------------------------------------------------------
    # DEFINE EXACT COMBINATIONS INSTEAD OF A GRID
    # Format: (n_workers, num_partitions, l, r)
    # ---------------------------------------------------------
    combinations = [
        # Infrastructure scaling (Fixed l=2, r=5)
        (2, 5, 2, 5),
        (4, 20, 2, 5),
        (8, 100, 2, 5),
        
        # Algorithmic OOD (Fixed workers=4, partitions=20)
        (4, 20, 5, 2),   # High l (your example)
        (4, 20, 1, 1),   # Greedy/Poor
        (4, 20, 2, 15),  # High iteration bottleneck
    ]
    
    results = []
    current_workers = None
    cluster, client = None, None
    current_partitions = None
    X_bag = None

    # Execute specific combinations
    for n_workers, num_partitions, l, r in combinations:
        
        if n_workers != current_workers:
            if client is not None:
                client.close()
                cluster.close()
            print(f"\n--- Starting Dask LocalCluster with {n_workers} workers ---")
            #cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
            client = Client('dask-scheduler:8786')
            current_workers = n_workers
            current_partitions = None 
            
        if num_partitions != current_partitions:
            X_bag = db.from_sequence(X_numpy, npartitions=num_partitions)
            current_partitions = num_partitions

        print(f"Testing: workers={n_workers}, partitions={num_partitions}, l={l}, r={r}")
        
        reset_counters()
        clf = kmeans_parallel(k=5, l=l, r=r)
        
        start_time = time.time()
        clf.compute_starting_centroids(X_bag, seed=42)
        clf.fit(X_bag, max_iter=10) 
        elapsed_time = time.time() - start_time
        
        cost = calculate_inertia(X_bag, clf.final_centroids)
        results.append({'workers': n_workers, 'parts': num_partitions, 'l': l, 'r': r, 'cost': cost, 'time': elapsed_time})
        print(f" -> Cost: {cost:.2f} | Time: {elapsed_time:.2f}s")
        
    if client is not None:
        client.close()
        cluster.close()
        
    print("\n--- Targeted Search Complete ---")
    for res in sorted(results, key=lambda x: x['cost']):
        print(res)

if __name__ == "__main__":
    main()