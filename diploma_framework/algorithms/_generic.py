import joblib, torch

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

        
