import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class VectorStore:
    
    #########################
    ##### Magic methods #####
    #########################
    
    def __init__(self, embed_size: int) -> None:
        """
        Initialize the VectorStore.
        """
        
        # Initialize the embeddings
        self.embeddings: np.ndarray = np.empty((0, embed_size), dtype=np.float32)
        
    
    #########################
    ##### Public methods ####
    #########################
        
    @staticmethod
    def load(path: str) -> 'VectorStore':
        """
        Load the embeddings from a file.
        
        Parameters
        ----------
        - path (str): The path to the file containing the embeddings.
        
        Returns
        -------
        - vector_store (VectorStore): The VectorStore object containing the embeddings.
        
        Raises
        ------
        - AssertionError: If the file does not exist.
        - AssertionError: If the embeddings are not a 2D array.
        """
        
        # Check if the file exists
        assert os.path.exists(path), f"File not found: {path}"
        
        # Load the embeddings
        embeddings = np.load(path)
        
        # Check if the embeddings are a 2D array
        assert embeddings.ndim == 2, "Embeddings must be a 2D array."
        
        # Create a new VectorStore
        vector_store = VectorStore(embed_size=embeddings.shape[1])
        
        # Set the embeddings
        vector_store.embeddings = embeddings
        
        return vector_store
        
        
    def save(self, path: str) -> None:
        """
        Save the embeddings to a file.
        
        Parameters
        ----------
        - path (str): The path to the file where the embeddings will be saved.
        """
        
        # Check if the embeddings are not None
        assert self.embeddings is not None, "No embeddings to save."
        
        # Save the embeddings
        np.save(path, self.embeddings)
        
        
    def update_at_indices(self, indices: np.ndarray, embeddings: np.ndarray) -> None:
        """
        Update the embeddings at the given indices.
        
        Parameters
        ----------
        - indices (np.ndarray): The indices to update.
        - embeddings (np.ndarray): The new embeddings to set.
        
        Raises
        ------
        - AssertionError: If the indices are not a list.
        - AssertionError: If the indices are not integers.
        - AssertionError: If the embeddings are not a 2D array.
        - AssertionError: If the embeddings do not have the same size as the embeddings in the store.
        """
        
        # Assert that the indices are a list of integers
        assert isinstance(indices, np.ndarray), "Indices must be an array of integers."
        assert all(isinstance(index, int) for index in indices), "Indices must be integers."
        assert len(indices) == len(set(indices)), "Indices must be unique."
        assert len(indices) == len(embeddings), "Indices and embeddings must have the same length."
        assert embeddings.ndim == 2, "Embeddings must be a 2D array."
        assert embeddings.shape[1] == self.embeddings.shape[1], "Embeddings must have the same size as the embeddings in the store."
        
        # Update the embeddings
        self.embeddings[indices] = embeddings
        
        
    def add_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Add embeddings to the store.
        
        Parameters
        ----------
        - embeddings (np.ndarray): The embeddings to add.
        
        Raises
        ------
        - AssertionError: If the embeddings are not a 2D array.
        - AssertionError: If the embeddings do not have the same size as the embeddings in the store.
        """
        
        # Assert that the embeddings are a 2D array
        assert embeddings.ndim == 2, "Embeddings must be a 2D array."
        assert embeddings.shape[1] == self.embeddings.shape[1], "Embeddings must have the same size as the embeddings in the store."
        
        # Add the embeddings
        self.embeddings = np.vstack([self.embeddings, embeddings])
        
        
    def count(self) -> int:
        """
        Get the number of embeddings in the store.
        
        Returns
        -------
        - count (int): The number of embeddings in the store.
        """
        
        return self.embeddings.shape[0]
    

    def search(self, query_embeddings: np.ndarray, k: int) -> tuple:
        """
        Search for the k most similar embeddings in the database to the query embedding.
        
        Parameters:
        -----------
        - embeddings (np.ndarray): The query embedding.
        - k (int): The number of most similar embeddings to return.
        
        Returns:
        --------
        - tuple: a tuple containing the sorted distances and indices of the k most similar embeddings
        """
        
        # Compute the cosine similarities between the query and database embeddings
        similarities = cosine_similarity(query_embeddings, self.embeddings).squeeze()
        
        # Convert the similarities to distances
        distances = 1.0 - similarities
        
        # Order the distances and get the indices of the k most similar
        sorted_indices = np.argsort(distances)[:k]
        sorted_distances = distances[sorted_indices]
        
        # Return the sorted distances and indices
        return sorted_distances.reshape(1, -1), sorted_indices.reshape(1, -1)