import os
import json
import faiss
import logging
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from typing import Optional, List, Dict
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer

from . import utils
from .cryptography import Cryptography


class SemanticDatabase:
    
    #########################
    ### Static attributes ###
    #########################
    
    DATABASE_FILE = "database.index" # Name of the vector database file
    DOCUMENTS_FILE = "documents.json" # Name of the documents mapping file
    
    QUERY_ENCODER_ID = "facebook/dpr-question_encoder-single-nq-base" # ID of the query encoder model
    DOCUMENTS_ECODER_ID = "facebook/dpr-ctx_encoder-single-nq-base" # ID of the documents encoder model
    
    
    #########################
    ##### Magic methods #####
    #########################
    
    def __init__(self, base_directory: Optional[str] = None):
        """
        Initialize the SemanticDatabase class.
        
        Parameters:
        -----------
        - base_directory (Optional[str]): the base directory to use. Default is the current working directory.
        """
        
        # Set the base directory
        self.base_directory = base_directory or os.getcwd()
        
        # Encoding models and tokenizers
        self.query_encoder: Optional[DPRQuestionEncoder] = None
        self.query_tokenizer: Optional[DPRQuestionEncoderTokenizer] = None
        self.documents_encoder: Optional[DPRContextEncoder] = None
        self.documents_tokenizer: Optional[DPRContextEncoderTokenizer] = None
        
        # Vector database and documents mapping
        self.vector_database: Optional[faiss.IndexIDMap] = None
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
        - Exception: if the database is not initialized
        """
        
        # Check if the database is initialized
        if not self._is_db_initialized():
            # Raise an exception if the database is not initialized
            raise Exception("The database is not initialized! Please load or create a database first.")
        
        # Embedding the query
        query_embedding = self._encode_texts([query_string], self.query_encoder, self.query_tokenizer) # type: ignore
        
        # Search for the most similar documents order by similarity
        # This returns a tuple with the distances and the indices of the documents
        distances, indices = self.vector_database.search(query_embedding, k=num_results) # type: ignore
        
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
            # Load the vector database
            self.vector_database = faiss.read_index(os.path.join(database_path, self.DATABASE_FILE))
            
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
            # Load the query and documents encoders and initialize the vector database and documents mapping
            logging.info("Creating a new semantic database...")
            self._load_models()
            self.documents_mapping = []
            self.vector_database = faiss.IndexIDMap(faiss.IndexFlatL2(self.documents_encoder.config.hidden_size)) # type: ignore
               
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
        - Exception: if the database is not initialized
        """
        
        # Check if the database is initialized
        if not self._is_db_initialized():
            # Raise an exception if the database is not initialized
            raise Exception("The database is not initialized! Please load or create a database first.")
        
        # Get the total number of documents to plot (Maximum 100)
        n_to_plot = min(100, self.vector_database.ntotal) # type: ignore
        
        # Compute the TSNE embeddings
        perplexity = min(30, n_to_plot - 1) # Set the perplexity to the minimum between 30 and the number of documents to plot
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        
        # Get the embeddings of the documents
        embeddings = np.array([self.vector_database.index.reconstruct(i) for i in range(n_to_plot)]) # type: ignore
        
        # Reduce the dimensionality of the embeddings to 2D
        embeddings_2d = tsne.fit_transform(embeddings) # type: ignore
        
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
            isinstance(self.vector_database, faiss.IndexIDMap) and
            isinstance(self.documents_mapping, list) and
            isinstance(self.documents_encoder, DPRContextEncoder) and
            isinstance(self.documents_tokenizer, DPRContextEncoderTokenizer) and
            isinstance(self.query_encoder, DPRQuestionEncoder) and
            isinstance(self.query_tokenizer, DPRQuestionEncoderTokenizer)
        )
    
    
    def _load_models(self) -> None:
        """
        Load the query and documents encoder models.
        """
        
        # Initialize the query and documents encoders
        self.query_encoder = DPRQuestionEncoder.from_pretrained(self.QUERY_ENCODER_ID)
        self.documents_encoder = DPRContextEncoder.from_pretrained(self.DOCUMENTS_ECODER_ID)
        
        # Initialize the query and documents tokenizers
        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(self.QUERY_ENCODER_ID)
        self.documents_tokenizer = DPRContextEncoderTokenizer.from_pretrained(self.DOCUMENTS_ECODER_ID)
        
        # Get the device
        self.device = utils.get_device()
        
        # Move the models to the device
        self.query_encoder.to(self.device) # type: ignore
        self.documents_encoder.to(self.device) # type: ignore

        
    def _save_db(self, destination_path: str) -> None:
        """
        Save the database to the specified path.
        
        Parameters:
        -----------
        - destination_path (str): the path where to save the database
        
        Raises:
        -------
        - Exception: if the destination path is not a folder
        """
        
        # Check if the destination path exists and is a folder
        if os.path.exists(destination_path) and not os.path.isdir(destination_path):
            # Raise an exception if the destination path is not a folder
            raise Exception("The destination path should be a folder!")
        
        # Create the destination folder if it does not exist
        os.makedirs(destination_path, exist_ok=True)
            
        # Save the vector database
        faiss.write_index(self.vector_database, os.path.join(destination_path, self.DATABASE_FILE))
        
        # Save the documents
        with open(os.path.join(destination_path, self.DOCUMENTS_FILE), "w") as f:
            json.dump(self.documents_mapping, f, indent=4)
    
    
    def _encode_texts(self, texts: list[str], encoder: DPRQuestionEncoder, tokenizer: DPRQuestionEncoderTokenizer, max_context_length: int = 512) -> np.ndarray:
        """
        Encode a list of texts using the specified encoder and tokenizer.
        
        Parameters:
        -----------
        - texts (list[str]): the list of texts to encode
        - encoder (DPRQuestionEncoder): the encoder model to use
        - tokenizer (DPRQuestionEncoderTokenizer): the tokenizer to use
        - max_context_length (int): the maximum length of the context
        
        Returns:
        --------
        - np.ndarray: the embeddings of the texts
        """
        
        # Tokenize the text and convert to PyTorch tensors
        inputs = tokenizer(
            texts, 
            return_tensors = "pt", 
            padding = True, 
            truncation = True,
            max_length = max_context_length 
        ).to(self.device)
        
        # Get the embeddings using the encoder model and convert them to a numpy array
        embeddings = encoder(**inputs).pooler_output.detach().cpu().numpy()
        
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
        """
        
        # Iterate over the files
        for document in tqdm(documents, desc="Embedding documents"):
            # Check if the file exists
            if not os.path.exists(document):
                # Log the warning and continue
                logging.warning(f"The document {document} does not exist! Skipping...")
                continue
            
            try:
                # Opening the document in read mode
                with open(document, 'r') as d:
                    # Reading the content of the document
                    document_content = d.read()
                    
                    # Computing the hash of the document content
                    document_hash = Cryptography.compute_hash(document_content)
                    
                    # Reading and encoding the content of the docuemnt into the embedding space
                    document_embedding = self._encode_texts([document_content], self.documents_encoder, self.documents_tokenizer) # type: ignore
                    
            except Exception as e:
                # Log the exception and continue to the next document
                logging.error(f"Error while processing the document {document}: {e}")
                continue
            
            # Get the relative path of the document
            rel_path = os.path.relpath(document, start=self.base_directory)
            
            # Look for an existing entry in the mapping based on the document path
            existing_entry = next((entry for entry in self.documents_mapping if entry.get("path") == rel_path), None)
            
            # Check if the document is already in the database
            if existing_entry is not None:
                # Check if the document hash is different from the existing one
                if existing_entry.get("hash") != document_hash:
                    # Log the information and update the document embedding
                    logging.info(f"Document {document} has changed. Updating its embedding...")
                    
                    # Document changed: update its embedding in the FAISS index.
                    idx = existing_entry["index"]
                    
                    try:
                        # Update the document embedding in the vector database
                        self._update_embedding_at_ids([idx], document_embedding[0])
                        
                        # Update the document hash in the mapping
                        existing_entry["hash"] = document_hash
                        
                    except Exception as e:
                        # Log the error and continue to the next document
                        logging.error(f"Error while updating embedding for {document}: {e}")
                        continue
            else:
                # Add the document embedding to the vector database
                self.vector_database.add_with_ids(document_embedding, np.array([self.vector_database.ntotal], dtype=np.int64)) # type: ignore
                
                # Append the document path to the documents mapping
                self.documents_mapping.append({
                    "index": self.vector_database.ntotal - 1, # type: ignore
                    "hash": document_hash,
                    "path": rel_path
                })
                
        # Return the embedded documents
        return self.documents_mapping
    
    
    def _update_embedding_at_ids(self, ids: list[int], documents_embeddings: list[np.ndarray]) -> None:
        """
        Update the embeddings of the documents at the specified indices.
        
        Parameters:
        -----------
        - ids (list[int]): the list of indices to update
        - documents_embeddings (list[np.ndarray]): the list of new document embeddings
        """
        
        # Convert the list of indices and embeddings to numpy arrays
        numpy_ids = np.array(ids, dtype=np.int64)
        numpy_embeddings = np.array(documents_embeddings)
        
        # Check if the embeddings are 1D
        if numpy_embeddings.ndim == 1:
            # Reshape the embeddings to 2D
            numpy_embeddings = numpy_embeddings.reshape(1, -1)
        
        # Remove the document at the specified index
        self.vector_database.remove_ids(numpy_ids) # type: ignore
        
        # Add the new document embedding at the specified index
        self.vector_database.add_with_ids(numpy_embeddings, numpy_ids) # type: ignore