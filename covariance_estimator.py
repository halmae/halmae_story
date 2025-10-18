import numpy as np

class CovarianceEstimator:
    def __init__(self,
                 window_size,
                 n_assets):
        
        self.n_assets = n_assets
        self.window_size = window_size
        self.return_sums = np.zeros(n_assets)
        self.return_sq_sums = np.zeros((n_assets, n_assets))
        self.sums_cnt = 0
        self.cov = np.zeros((n_assets, n_assets))

    
    def calculate_covariance(self, T, return_data):
        if T < self.window_size:
            self.window_update(return_data[T], 1)
            if T == self.window_size - 1:
                return self.compute_cov_matrix()
            return None
        
        self.window_update(return_data[T], 1)
        self.window_update(return_data[T - self.window_size], -1)
        return self.compute_cov_matrix()
    
    def compute_cov_matrix(self):
        if self.sums_cnt < 2:
            return None
        
        mean = self.return_sums / self.sums_cnt

        for ii in range(self.n_assets):
            for jj in range(ii, self.n_assets):
                cov_val = (self.return_sq_sums[ii][jj] / self.sums_cnt - mean[ii] * mean[jj])
                cov_val *= self.sums_cnt / (self.sums_cnt - 1)

                self.cov[ii, jj] = cov_val
                if ii != jj:
                    self.cov[jj, ii] = cov_val

        return self.cov.copy()
    

    def window_update(self, return_data, direction):
        assert direction in [1, -1], "direction must be 1 or -1"

        for ii in range(self.n_assets):
            self.return_sums[ii] += return_data[ii] * direction
            for jj in range(ii, self.n_assets):
                self.return_sq_sums[ii, jj] += (return_data[ii] * return_data[jj] * direction)

                if ii != jj:
                    self.return_sq_sums[jj, ii] = self.return_sq_sums[ii, jj]

        self.sums_cnt += direction

        