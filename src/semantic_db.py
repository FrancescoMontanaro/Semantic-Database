import os
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from transformers import AutoConfig
from FlagEmbedding import BGEM3FlagModel
from typing import Optional, List, Dict, Any

from . import utils
from .cryptography import Cryptography


class SemanticDatabase:
    
    #########################
    ### Static attributes ###
    #########################
    
    EMBEDDINGS_FILE = "database.npy" # Name of the embeddings file
    DOCUMENTS_FILE = "documents.json" # Name of the documents mapping file
    MODEL_ID = "BAAI/bge-m3" # ID of the encoder model
    
    
    #########################
    ##### Magic methods #####
    #########################
    
    def __init__(self, base_directory: Optional[str] = None, batch_size: int = 1, context_length: int = 8192) -> None:
        """
        Initialize the SemanticDatabase class.
        
        Parameters:
        -----------
        - base_directory (Optional[str]): the base directory to use. Default is the current working directory.
        - batch_size (int): the batch size to use for embedding the documents
        - context_length (int): the maximum length of the context
        """
        
        # Save the parameters
        self.base_directory = base_directory or os.getcwd()
        self.batch_size = batch_size
        self.context_length = context_length
        
        # Encoding model
        self.encoder: Optional[BGEM3FlagModel] = None
        
        # Vector database and documents mapping
        self.embeddings_db: Optional[np.ndarray] = None
        self.documents_mapping: List[Dict] = []
        
       
    #########################
    ##### Public methods ####
    #########################
    
    def query(self, query_string: str, num_results: int = 1) -> list[dict]:
        """
        Perform a query to the database and return the most semantically similar documents.
        
        Parameters:
        -----------
        - query_string (str): the query string
        - num_results (int): the number of results to return
        
        Returns:
        --------
        - list[dict]: a list of dictionaries containing the index, document path and distance of the most similar documents
        
        Raises:
        -------
        - AssertionError: if the database is not initialized or the encoder is not defined
        """
        
        # Check if the encoder is defined
        assert self._is_db_initialized(), "The database is not initialized! Please load or create a database first!"
        assert isinstance(self.encoder, BGEM3FlagModel), "The encoder should be defined!"
        assert isinstance(self.embeddings_db, np.ndarray), "The embeddings database should be initialized!"
        
        # Embedding the query
        query_embedding = self._encode_texts([query_string], self.encoder)
        
        # Search for the most similar documents order by similarity
        # This returns a tuple with the distances and the indices of the documents
        distances, indices = self._search(
            query_embedding = query_embedding,
            db_embeddings = self.embeddings_db,
            k = num_results
        )
        
        # Initialize the results list
        results = []
        
        # Iterate over the distances and indices
        for dist, idx in zip(distances[0], indices[0]):
            # Extract the document from the documents mapping
            target_document = next((doc for doc in self.documents_mapping if doc.get("index") == idx), None)
            
            # Check if the document exists
            if target_document is None:
                # Log the warning and continue
                logging.error(f"Document with index {idx} not found! Skipping...")
                continue
            
            # Check if the target document exists in the base directory
            if not os.path.exists(os.path.join(self.base_directory, target_document["path"])):
                # Log a warning
                logging.warning(f"Document {target_document['path']} does not exist in the base directory. Maybe it was moved or the base directory has changed.")
            
            # Append the document to the results list
            results.append({
                "index": int(idx),
                "distance": float(dist),
                "relative_path": target_document.get("path"),
                "absolute_path": os.path.join(self.base_directory, target_document["path"])
            })
            
        # Return the results
        return results
    
    
    def load(self, database_path: str) -> None:
        """
        Load an existing semantic database from the specified path.
        
        Parameters:
        -----------
        - database_path (str): the path of the database to load
        - query_encoder_id (str): the ID of the query encoder model
        - documents_encoder_id (str): the ID of the documents encoder model
        
        Raises:
        -------
        - Exception: if the database path does not exist
        - Exception: if the configuration file is invalid
        - Exception: if the vector database file is invalid
        - Exception: if the documents mapping file is invalid
        """
        
        # Log status
        logging.info("Loading the semantic database...")
        
        # Check if the database path exists
        if not os.path.exists(database_path):
            # Raise an exception if the database path does not exist
            raise Exception("The specified database path does not exist!")
        
        # Loading the models if not already loaded
        if not self._is_db_initialized():
            # Load the query and documents encoders
            self._load_models()
        
        try:
            # Load the embeddings database
            self.embeddings_db = np.load(os.path.join(database_path, self.EMBEDDINGS_FILE))
            
        except Exception as e:
            # Raise an exception if the vector database file is invalid
            raise Exception(f"Invalid vector database file: {e}")
        
        try:
            # Load the documents mapping
            with open(os.path.join(database_path, self.DOCUMENTS_FILE), "r") as f:
                self.documents_mapping = json.load(f)
                
        except Exception as e:
            # Raise an exception if the documents mapping file is invalid
            raise Exception(f"Invalid documents mapping file: {e}")
        
        # Log status
        logging.info("Semantic database loaded successfully!")
    
    
    def create_or_update(self, destination_path: Optional[str] = None) -> None:
        """
        Create a semantic database from the documents in the specified path.
        
        Parameters:
        -----------
        - destination_path (Optional[str]): the path where to save the database
        
        Raises:
        -------
        - Exception: if the destination path is not a folder
        """
        
        # Check if the database is initialized
        if not self._is_db_initialized():
            # Log status
            logging.info("Creating a new semantic database...")
            
            # Load the model and extract the hidden size
            embed_size = self._load_models()
            
            # Initialize the vector database and documents mapping
            self.documents_mapping = []
            self.embeddings_db = np.empty((0, embed_size), dtype=np.float32)
               
        # List the documents in the specified path
        logging.info("Listing the documents in the specified path...")
        documents = utils.list_files(self.base_directory)
        
        # Embed the documents
        self._embed_documents(documents)
                
        # Check if the destination path is specified
        if destination_path is not None:
            # Save the database to the specified path
            logging.info("Saving the semantic database to the specified path...")
            self._save_db(destination_path)
            
        # Log status
        logging.info("Semantic database created successfully!")
    
    
    def plot_space(self) -> None:
        """
        Reproject the embedding space in 2D using TSNE and plot the documents.
        
        Raises:
        -------
        - AssertionError: if the database is not initialized or the encoder is not defined
        """
        
        # Check if the vector database is an instance of faiss.IndexIDMap
        assert self._is_db_initialized(), "The database is not initialized! Please load or create a database first!"
        assert isinstance(self.embeddings_db, np.ndarray), "The embeddings database should be initialized!"
        
        # Get the total number of documents to plot (Maximum 100)
        n_to_plot = min(100, self.embeddings_db.shape[0])
        
        # Compute the TSNE embeddings
        perplexity = min(30, n_to_plot - 1) # Set the perplexity to the minimum between 30 and the number of documents to plot
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        
        # Reduce the dimensionality of the embeddings to 2D
        embeddings_2d = tsne.fit_transform(self.embeddings_db[:n_to_plot])
        
        # Plot the 2D embeddings
        plt.figure(figsize=(8, 6))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c="b", edgecolor="k")
        
        # Add labels to the points
        for i, txt in enumerate(self.documents_mapping[:n_to_plot]):
            plt.annotate(txt["path"], (embeddings_2d[i, 0], embeddings_2d[i, 1]))
            
        # Show the plot
        plt.title("2D embeddings of the documents")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.grid(True)
        plt.show()
    
    
    #########################
    ### Protected methods ###
    #########################
    
    def _is_db_initialized(self) -> bool:
        """
        Check if the database is initialized
        
        Returns:
        --------
        - bool: True if the database is initialized, False otherwise
        """
        
        # Check if the database is initialized by checking the types of the variables
        return (
            isinstance(self.embeddings_db, np.ndarray) and
            isinstance(self.documents_mapping, list) and
            self.encoder is not None
        )
    
    
    def _load_models(self) -> int:
        """
        Load the query and documents encoder models.
        
        Returns:
        --------
        - int: the hidden size of the encoder model
        """

        # Initialize the query and documents encoders
        self.encoder = BGEM3FlagModel(self.MODEL_ID, use_fp16=True)
        
        # Get the hidden size of the encoder model
        config = AutoConfig.from_pretrained(self.MODEL_ID)
        
        # Return the hidden size of the encoder model
        return getattr(config, "hidden_size")

        
    def _save_db(self, destination_path: str) -> None:
        """
        Save the database to the specified path.
        
        Parameters:
        -----------
        - destination_path (str): the path where to save the database
        
        Raises:
        -------
        - AssertionError: if the embeddings database is not initialized
        - AssertionError: if the destination path is not a folder
        """
        
        # Check if the embeddings database is initialized and the destination path is a folder
        assert isinstance(self.embeddings_db, np.ndarray), "The embeddings database should be initialized!"
        assert not (os.path.exists(destination_path) and not os.path.isdir(destination_path)), "The destination path should be a folder!"
        
        # Create the destination folder if it does not exist
        os.makedirs(destination_path, exist_ok=True)
            
        # Save the vector database
        np.save(os.path.join(destination_path, self.EMBEDDINGS_FILE), self.embeddings_db)
        
        # Save the documents
        with open(os.path.join(destination_path, self.DOCUMENTS_FILE), "w") as f:
            json.dump(self.documents_mapping, f, indent=4)
    
    
    def _encode_texts(self, texts: list[str], encoder: Any) -> np.ndarray:
        """
        Encode a list of texts using the specified encoder and tokenizer.
        
        Parameters:
        -----------
        - texts (list[str]): the list of texts to encode
        - encoder (DPRQuestionEncoder): the encoder model to use
        - tokenizer (DPRQuestionEncoderTokenizer): the tokenizer to use
        
        Returns:
        --------
        - np.ndarray: the embeddings of the texts
        """
        
        # Get the embeddings using the encoder model and convert them to a numpy array
        embeddings = encoder.encode(texts, batch_size=self.batch_size, max_length=self.context_length)['dense_vecs']
        
        # Return the embeddings
        return embeddings
    
    
    def _embed_documents(self, documents: list[str]) -> list[dict]:
        """
        Embed the documents and add them to the vector database.
        
        Parameters:
        -----------
        - documents (list[str]): the list of documents paths to embed
        
        Returns:
        --------
        - list[dict]: the list of embedded documents
        
        Raises:
        -------
        - AssertionError: if the encoder is not defined or the embeddings database is not initialized
        """
        
        # Check if the encoder is defined
        assert isinstance(self.encoder, BGEM3FlagModel), "The encoder should be defined!"
        assert isinstance(self.embeddings_db, np.ndarray), "The embeddings database should be initialized!"
        
        # Iterate over the documents
        idx = 0
        while idx < len(documents):
            
            # Iterate untile the batch size to take precise batches of documents
            batch = []
            while len(batch) < self.batch_size and idx < len(documents):
                # Extract the document
                document = documents[idx]
            
                # Check if the file exists
                if not os.path.exists(document):
                    # Log the warning and continue
                    logging.warning(f"The document {document} does not exist! Skipping...")
                    idx += 1
                    continue
                
                try:
                    # Opening the document in read mode
                    with open(document, 'r') as d:
                        # Reading the content of the document
                        document_content = d.read()
                        
                    # Extract the relative path of the document and compute its hash
                    document_rel_path = os.path.relpath(document, start=self.base_directory)
                    document_hash = Cryptography.compute_hash(document_content)
                    
                    # Look for an existing entry in the mapping based on the document path
                    existing_entry = next((entry for entry in self.documents_mapping if entry.get("path") == document_rel_path), None)
                    
                    # Check if the document is already in the database
                    entry_to_update_idx = None
                    if existing_entry is not None:
                        # Check if the document hash is different from the existing one
                        if existing_entry.get("hash") != document_hash:
                            # Log the information and update the document embedding
                            logging.info(f"Document {document} has changed. Updating its embedding...")
                    
                            # Document changed: Save the index of the document to update in the FAISS index
                            entry_to_update_idx = existing_entry["index"]
                        
                        else:
                            # Log the information and continue to the next document
                            logging.info(f"Document {document} is already in the database. Skipping...")
                            idx += 1
                            continue
                    
                    # Append the document content to the list
                    batch.append({
                        "hash": document_hash,
                        "path": document_rel_path,
                        "content": document_content,
                        "entry_to_update_idx": entry_to_update_idx
                    })
                        
                except Exception as e:
                    # Log the exception and continue to the next document
                    logging.error(f"Error while processing the document {document}: {e}")
                    continue
                
                finally:
                    # Increment the index
                    idx += 1
            
            # Check if the batch is empty
            if len(batch) == 0:
                # Break the loop if the batch is empty
                break
            
            # Embed the batch of documents
            batch_embeddings = self._encode_texts(
                texts = [doc["content"] for doc in batch],
                encoder = self.encoder
            )
            
            # Iterate over the documents and corresponding embeddings
            for document, embedding in zip(batch, batch_embeddings):
                # Check if the document is already in the database
                if document["entry_to_update_idx"] is not None:
                    # Update the embedding of the document in the vector database
                    self.embeddings_db[document["entry_to_update_idx"], :] = embedding
                    
                    # Update the document hash in the mapping
                    self.documents_mapping[document["entry_to_update_idx"]]["hash"] = document["hash"]
                    
                else:
                    # Add the document embedding to the vector database
                    self.embeddings_db = np.vstack([self.embeddings_db, embedding.reshape(1, -1)])
                    
                    # Append the document path to the documents mapping
                    self.documents_mapping.append({
                        "index": self.embeddings_db.shape[0] - 1,
                        "hash": document["hash"],
                        "path": document["path"]
                    })
                    
                # Log the information
                logging.info(f"Embedded document --> {document['path']}")
                
        # Return the embedded documents
        return self.documents_mapping
        
        
    @staticmethod
    def _search(query_embedding: np.ndarray, db_embeddings: np.ndarray, k: int) -> tuple:
        """
        Search for the k most similar embeddings in the database to the query embedding.
        
        Parameters:
        -----------
        - query_embedding (np.ndarray): the query embedding
        - db_embeddings (np.ndarray): the database embeddings
        
        Returns:
        --------
        - tuple: a tuple containing the sorted distances and indices of the k most similar embeddings
        """
        
        # Normalize the query and database embeddings by their L2 norm
        query_norm = query_embedding / (np.linalg.norm(query_embedding, axis=1, keepdims=True) + 1e-10)
        db_norm = db_embeddings / (np.linalg.norm(db_embeddings, axis=1, keepdims=True) + 1e-10)
        
        # Compute the cosine similarities between the query and database embeddings
        similarities = np.dot(db_norm, query_norm.T).squeeze()
        
        # Convert the similarities to distances
        distances = 1.0 - similarities
        
        # Order the distances and get the indices of the k most similar
        sorted_indices = np.argsort(distances)[:k]
        sorted_distances = distances[sorted_indices]
        
        # Return the sorted distances and indices
        return sorted_distances.reshape(1, -1), sorted_indices.reshape(1, -1)