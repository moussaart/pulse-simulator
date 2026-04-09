from .Localization_alghorthime import LocalizationAlgorthimes

class Alghortimes_doc():
    
    _cached_algorithms = None

    def __init__(self):
        self.algorithm_methods = {
            "Trilateration": LocalizationAlgorthimes.trilateration,
            "NLOS-Aware AEKF": LocalizationAlgorthimes.Nlos_aware_aekf,
            "IMU Only": LocalizationAlgorthimes.imu_only_filter,
        }
        
        # Load custom algorithms only once
        if Alghortimes_doc._cached_algorithms is None:
            import os
            from src.utils.resource_loader import get_data_path
            user_algo_dir = get_data_path(os.path.join("src", "user_algorithms"))
            
            from .algorithm_loader import AlgorithmLoader
            loader = AlgorithmLoader(user_algo_dir)
            Alghortimes_doc._cached_algorithms = loader.discover_algorithms()
        
        # Merge custom algorithms from cache
        self.algorithm_methods.update(Alghortimes_doc._cached_algorithms)

    @staticmethod
    def reload_custom_algorithms():
        """Force reload of custom algorithms from disk"""
        Alghortimes_doc._cached_algorithms = None
        
        import os
        from src.utils.resource_loader import get_data_path
        user_algo_dir = get_data_path(os.path.join("src", "user_algorithms"))
        
        from .algorithm_loader import AlgorithmLoader
        loader = AlgorithmLoader(user_algo_dir)
        Alghortimes_doc._cached_algorithms = loader.discover_algorithms()

    def get_algorithm_methods(self):
        return self.algorithm_methods
