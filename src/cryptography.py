import hashlib

class Cryptography:
    """
    Class that provides cryptographic functionalities.
    
    Methods:
    --------
    - compute_hash(data: str) -> str: Compute the hash of the given data.
    - verify_hash(data: str, hash: str) -> bool: Verify whether the given hash is correct for the given data.
    """
    
    ######################
    ### Public methods ###
    ######################
    
    @staticmethod
    def compute_hash(data: str) -> str:
        """
        Compute the hash of the given data.
        
        Parameters:
        -----------
        - data (str): The data to hash.
        
        Returns:
        --------
        - str: The hash of the given data.
        """
        
        # Compute the hash of the data
        return hashlib.sha256(data.encode()).hexdigest()
    
    
    @staticmethod
    def verify_hash(data: str, hash: str) -> bool:
        """
        Verify whether the given hash is correct for the given data.
        
        Parameters:
        -----------
        - data (str): The data to hash.
        - hash (str): The hash to verify.
        
        Returns:
        --------
        - bool: True if the hash is correct, False otherwise.
        """
        
        # Compare the hash of the data with the given hash
        return hashlib.sha256(data.encode()).hexdigest() == hash
