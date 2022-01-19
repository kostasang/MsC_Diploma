from typing import Tuple
from diploma_framework.evaluation import test_env
import joblib, torch, numpy as np

class DeepRLAlgorithm():

    """
    Implements basic methods that every algorithm in the framework will inherit

    """

    def get_model(self) -> torch.nn.Module :
        
        """
        Returns th algorithms neural network model

        """
        return self.model 

    def save_model(self, path : str) -> None :

        """
        Saves the algorithm's model to the specified path

        """
        try:
            joblib.dump(self.model, path)
        except:
            raise Exception(f'Poblem storing model to {path}')

    def evaluate(self, n_evaluations : int) -> Tuple[float, float] :

        """
        Evaluate model on environment n_evaluations times and 
        extract average metrics

        """

        evaluations = [test_env(self.env, self.model, vis=False) for _ in range(n_evaluations)]
        average_reward = np.mean([metric[0] for metric in evaluations])
        average_nframes = np.mean([metric[1] for metric in evaluations])
        
        return average_reward, average_nframes