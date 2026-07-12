dist_comp_times = 0

def reset_counters():
    global dist_comp_times
    dist_comp_times = 0



class kmeans_parallel(): 

#--------------------------------------------------------------------------------------

    def __init__(self, k, l, r):
        self.k = k
        self.l = l #oversampling factor
        self.r = r #number of iterations
        self.centroids = []

#--------------------------------------------------------------------------------------

    def compute_starting_centroids(self, X, alpha=1, l=None, max_iter=None, seed=None):
        import dask
        import dask.bag as db
        import random
        from sklearn.cluster import KMeans
        import numpy as np
        from dask.bag import random as db_random
        global dist_comp_times

        if seed is not None:
            np.random.seed(seed)
        ######################################################
        # STEP 1
        
        initial_centroid = db.random.sample(X,1).compute()[0]
        initial_centroid = np.asarray(initial_centroid).reshape(1, -1)
        self.centroids.append(initial_centroid)

        c0 = initial_centroid[0]
        state = X.map(lambda x: (np.linalg.norm(x - c0) ** 2, 0)) # bag with (min_distance from clusters, nearest cluster) --> here (dist_c0,c0)

        # we need to persist because the value is needed for further computations
        state = dask.persist(state)[0]

        # incrementing counters
        dist_comp_times += 1
        
        ################################################
        # STEP 2
        
        psi = state.map(lambda t: t[0]).sum().compute()

        #####################################################
        # STEP 3
        
        if l is None:
            l = self.l
        if max_iter is None:
            max_iter = self.r if self.r is not None else int(round(alpha * np.log(psi)))

        # print("Number of iterations:", max_iter)

        for _ in range(max_iter):
            
            # evaluate cost of the current clusters for the iteration
            cost = state.map(lambda t: t[0]).sum().compute()
            
            # evaluates probabilities for the points
            probs = state.map(lambda t: min(1.0, t[0] * l / cost))

            # pairing points and probabilities
            paired = db.zip(X, probs)
            # sampling from this distribution with the chosen probability (we generate for each value a random uniform value, and if it is lower than the probablity associated to the point, it gets sampled.
            sample = (
                paired.filter(lambda t: np.random.uniform() < t[1])
                      .map(lambda t: t[0])
            ).compute()

            # start new iteration if we don't find any samples 
            if not sample:
                continue

            start_idx = len(self.centroids)
            new_centroids_arr = np.vstack([np.asarray(s).reshape(1, -1) for s in sample])
            
            for s in sample:
                self.centroids.append(np.asarray(s).reshape(1, -1))

            # Compute distances of points to new centroids, if they are lower than the old ones, change state   
            def update_state(pack, new_arr=new_centroids_arr, s_idx=start_idx):
                x, (curr_min_dist, curr_idx) = pack
                dists_to_new = np.linalg.norm(x - new_arr, axis=1) ** 2
                min_new_idx_relative = int(np.argmin(dists_to_new))
                min_new_dist = dists_to_new[min_new_idx_relative]
                
                if min_new_dist < curr_min_dist:
                    return (min_new_dist, s_idx + min_new_idx_relative)
                else:
                    return (curr_min_dist, curr_idx)
            
            state = db.zip(X, state).map(update_state)
            state = dask.persist(state)[0]

            dist_comp_times += 1

        ###################################################
        # STEP 7
        # Extract weights directly from the optimally tracked state
        
        counts = state.map(lambda t: t[1]).frequencies().compute()
        weights = np.zeros(len(self.centroids))
        
        for center_idx, count in counts:
            weights[center_idx] = count

        centroids_weights = weights 

        ###################################################
        # STEP 8 using scikit kmeans 
        
        kmeans = KMeans(n_clusters=self.k)
        kmeans.fit(np.vstack(self.centroids), sample_weight=centroids_weights)
        self.starting_centroids = kmeans.cluster_centers_

        self.min_dists = state.map(lambda t: t[0])

#--------------------------------------------------------------------------------------

    def fit(self, X, max_iter=100, tol=1e-4):
        
        '''
        Standard K-means implementation after the parallel algorithm initialization 
        '''
        
        # Starting point for Lloyd's iteration
        centroids_arr = np.vstack(self.starting_centroids)

        for _ in range(max_iter):
            # Map each point to its closest centroid: (cluster_index, point)
            mapped = X.map( lambda x: (int(np.argmin(np.linalg.norm(x - centroids_arr, axis=1))), np.array(x)) )
            
            # Group by cluster index and compute the average of the points
            def binop(acc, item):
                total, count = acc
                return (total + item[1], count + 1)
            
            def combine(acc1, acc2):
                return (acc1[0] + acc2[0], acc1[1] + acc2[1])
            
            sums_counts = mapped.foldby(lambda t: t[0], binop, (0.0, 0), combine, (0.0, 0)).compute()
            reduced = {idx: total/count for idx, (total, count) in sums_counts}
            
            # Update the centroids with the new computed averages
            new_centroids = np.copy(centroids_arr)
            for cluster_idx, p_mean in reduced:
                new_centroids[cluster_idx] = p_mean
            
            # keep the latest result
            self.final_centroids = new_centroids

            # Check for convergence 
            if np.linalg.norm(new_centroids - centroids_arr) < tol:
                break

            centroids_arr = new_centroids
            
#--------------------------------------------------------------------------------------

    def classify(self, X):
        centroids_arr = np.vstack(self.final_centroids)
        # Returns a Dask bag mapping each point to its closest cluster index
        return X.map(lambda x: int( np.argmin(np.linalg.norm(x - centroids_arr, axis=1))) )

#--------------------------------------------------------------------------------------
