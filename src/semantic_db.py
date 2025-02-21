import os
import json
import faiss
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer

from . import utils


class SemanticDatabase:
    
    ### Static attributes ###
    
    config_file = "config.json" # Name of the database configuration file
    database_file = "database.index" # Name of the vector database file
    documents_file = "documents.json" # Name of the documents mapping file 
    
    ### Magic methods ###
    
    def __init__(self) -> None:
        """
        Initialize the SemanticDatabase class.
        """
        
        # Create the query and documents encoders and tokenizers
        self.query_encoder, self.query_tokenizer = None, None
        self.documents_encoder, self.documents_tokenizer = None, None
        
        # Create a variable to store the vector database and documents mapping
        self.vector_database = None
        self.documents_mapping = []
        
        
    ### Public methods ###
    
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
        - Exception: if the database is not consistent
        """
        
        # Check if the database is initialized
        if not self._is_database_initialized():
            # Raise an exception if the database is not initialized
            raise Exception("The database is not initialized! Please load or create a database first.")
        
        # Check if the database is consistent
        if not self._is_database_consistent():
            # Raise an exception if the database is not consistent
            raise Exception("The database is not consistent! The mapping between documents and embeddings is not correct.")
        
        # Embedding the query
        query_embedding = self._encode_texts([query_string], self.query_encoder, self.query_tokenizer) # type: ignore
        
        # Search for the most similar documents order by similarity
        # This returns a tuple with the distances and the indices of the documents
        distances, indices = self.vector_database.search(query_embedding, k=num_results) # type: ignore
        
        # Initialize the results list
        results = []
        
        # Iterate over the distances and indices
        for dist, idx in zip(distances[0], indices[0]):
            # Append the results to the list
            results.append({
                "index": int(idx),
                "document": self.documents_mapping[idx],
                "distance": float(dist)
            })
            
        # Return the results
        return results
    
    
    def update(self, documents_path: str, destination_path: Optional[str] = None) -> None:
        """
        Update the semantic database with new documents.
        
        Parameters:
        -----------
        - documents_path (str): the path of the documents to update
        - destination_path (Optional[str]): the path where to save the updated database
        
        Raises:
        -------
        - Exception: if the database is not initialized
        - Exception: if the database is not consistent
        - Exception: if the number of embedded documents is not equal to the number of documents
        """
              
        # Log status
        logging.info("Updating the semantic database with new documents...")
        
        # Check if the database is initialized
        if not self._is_database_initialized():
            # Raise an exception if the database is not initialized
            raise Exception("The database is not initialized! Please load or create a database first.")
        
        # Check if the database is consistent
        if not self._is_database_consistent():
            # Raise an exception if the database is not consistent
            raise Exception("The database is not consistent! The mapping between documents and embeddings is not correct.")
        
        # Log status
        logging.info("Listing the documents in the specified path...")
        
        # List the documents in the specified path
        documents = utils.list_files(documents_path)
        
        # Log status
        logging.info("Embedding the new documents...")
        
        # Embed the documents
        embedded_documents = self._embed_documents(documents)
        
        # Append the new documents to the documents mapping
        self.documents_mapping.extend(embedded_documents)
        
        # Check if the number of embedded documents is equal to the number of documents
        if not self._is_database_consistent():
            # Raise an exception if the number of embedded documents is not equal to the number of documents
            raise Exception("Inconsistent database update! The number of embedded documents is not equal to the number of documents!")
        
        # Check if the destination path is specified
        if destination_path is not None:
            # Log status
            logging.info("Saving the updated semantic database to the specified path...")
            
            # Overwrite the database to the specified path
            faiss.write_index(self.vector_database, os.path.join(destination_path, self.database_file))
        
            # Overwrite the documents mapping
            with open(os.path.join(destination_path, self.documents_file), "w") as f:
                json.dump(self.documents_mapping, f, indent=4)
                
        # Log status
        logging.info("Semantic database updated successfully!")
    
    
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
        - Exception: if the database is not consistent
        """
        
        # Log status
        logging.info("Loading the semantic database...")
        
        # Check if the database path exists
        if not os.path.exists(database_path):
            # Raise an exception if the database path does not exist
            raise Exception("The specified database path does not exist!")
        
        try:
            # Load the configuration file
            with open(os.path.join(database_path, self.config_file), "r") as f:
                config = json.load(f)
                
            # Load the query and documents encoders
            self._load_models(config["query_encoder_id"], config["documents_encoder_id"])
            
        except Exception as e:
            # Raise an exception if the configuration file is invalid
            raise Exception(f"Invalid database configuration file: {e}")
        
        try:
            # Load the vector database
            self.vector_database = faiss.read_index(os.path.join(database_path, self.database_file))
            
        except Exception as e:
            # Raise an exception if the vector database file is invalid
            raise Exception(f"Invalid vector database file: {e}")
        
        try:
            # Load the documents mapping
            with open(os.path.join(database_path, self.documents_file), "r") as f:
                self.documents_mapping = json.load(f)
                
        except Exception as e:
            # Raise an exception if the documents mapping file is invalid
            raise Exception(f"Invalid documents mapping file: {e}")
        
        # Check if the database is consistent
        if not self._is_database_consistent():
            # Raise an exception if the database is not consistent
            raise Exception("Error loading the database! The database is not consistent since the mapping between documents and embeddings is not correct.")
        
        # Log status
        logging.info("Semantic database loaded successfully!")
    
    
    def create(self, documents_path: str, query_encoder_id: str, documents_encoder_id: str, destination_path: Optional[str] = None) -> None:
        """
        Create a semantic database from the documents in the specified path.
        
        Parameters:
        -----------
        - documents_path (str): the path of the documents to embed
        - query_encoder_id (str): the ID of the query encoder model
        - documents_encoder_id (str): the ID of the documents encoder model
        - destination_path (Optional[str]): the path where to save the database
        
        Raises:
        -------
        - Exception: if the destination path is not a folder
        - Exception: if the number of embedded documents is not equal to the number of documents
        """
        
        # Check if the destination path is specified and is a folder
        if destination_path is not None and os.path.exists(destination_path) and not os.path.isdir(destination_path):
            # Raise an exception if the destination path is not a folder
            raise Exception("The destination path should be a folder!")
        
        # Log status
        logging.info("Loading the query and documents encoders...")
        
        # Load the query and documents encoders
        self._load_models(query_encoder_id, documents_encoder_id)
        
        # Log status
        logging.info("Listing the documents in the specified path...")
        
        # List the documents in the specified path
        documents = utils.list_files(documents_path)
        
        # Log status
        logging.info("Creating the semantic database...")
        
        # Initializing the vector database with the ebedding dimensions 
        self.vector_database = faiss.IndexFlatL2(self.documents_encoder.config.hidden_size) # type: ignore
        
        # Log status
        logging.info("Embedding the documents...")
        
        # Embed the documents
        embedded_documets = self._embed_documents(documents)
            
        # Check if the number of embedded documents is equal to the number of documents
        if len(embedded_documets) != self.vector_database.ntotal:
            # Raise an exception if the number of embedded documents is not equal to the number of documents
            raise Exception("Inconsistent database creation! The number of embedded documents is not equal to the number of documents!")
                
        # Save the documents mapping
        self.documents_mapping = embedded_documets
                
        # Check if the destination path is specified
        if destination_path is not None:
            # Log status
            logging.info("Saving the semantic database to the specified path...")
            
            # Save the database to the specified path
            self._save_database(
                destination_path = destination_path, 
                query_encoder_id = query_encoder_id, 
                documents_encoder_id = documents_encoder_id
            )
            
        # Log status
        logging.info("Semantic database created successfully!")
    
    
    ### Protected methods ###
    
    def _is_database_initialized(self) -> bool:
        """
        Check if the database is initialized
        
        Returns:
        --------
        - bool: True if the database is initialized, False otherwise
        """
        
        # Check if the database is initialized by checking the types of the variables
        return (
            isinstance(self.vector_database, faiss.IndexFlatL2) and
            isinstance(self.documents_mapping, list) and
            isinstance(self.documents_encoder, DPRContextEncoder) and
            isinstance(self.documents_tokenizer, DPRContextEncoderTokenizer) and
            isinstance(self.query_encoder, DPRQuestionEncoder) and
            isinstance(self.query_tokenizer, DPRQuestionEncoderTokenizer)
        )
        
    
    def _is_database_consistent(self) -> bool:
        """
        Check if the database is consistent
        
        Returns:
        --------
        - bool: True if the database is consistent, False otherwise
        """
        
        # Check if the number of documents in the mapping is equal to the number of documents in the vector database
        return len(self.documents_mapping) == self.vector_database.ntotal # type: ignore
    
    
    def _load_models(self, query_encoder_id: str, documents_encoder_id: str) -> None:
        """
        Load the query and documents encoder models.
        
        Parameters:
        -----------
        - query_encoder_id (str): the ID of the query encoder model
        - documents_encoder_id (str): the ID of the documents encoder model
        """
        
        # Initialize the query and documents encoders
        self.query_encoder = DPRQuestionEncoder.from_pretrained(query_encoder_id)
        self.documents_encoder = DPRContextEncoder.from_pretrained(documents_encoder_id)
        
        # Initialize the query and documents tokenizers
        self.query_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(query_encoder_id)
        self.documents_tokenizer = DPRContextEncoderTokenizer.from_pretrained(documents_encoder_id)
        
        # Get the device
        self.device = utils.get_device()
        
        # Move the models to the device
        self.query_encoder.to(self.device) # type: ignore
        self.documents_encoder.to(self.device) # type: ignore
        
        
    def _save_database(self, destination_path: str, query_encoder_id: str, documents_encoder_id: str) -> None:
        """
        Save the database to the specified path.
        
        Parameters:
        -----------
        - destination_path (str): the path where to save the database
        - query_encoder_id (str): the ID of the query encoder model
        - documents_encoder_id (str): the ID of the documents encoder model
        """
        
        # Create the destination folder if it does not exist
        os.makedirs(destination_path, exist_ok=True)
        
        # Save the configuration file
        with open(os.path.join(destination_path, self.config_file), "w") as f:
            # Create a configuration file to store the encoder IDs
            json.dump({
                "query_encoder_id": query_encoder_id,
                "documents_encoder_id": documents_encoder_id
            }, f, indent=4)
            
        # Save the vector database
        faiss.write_index(self.vector_database, os.path.join(destination_path, self.database_file))
        
        # Save the documents
        with open(os.path.join(destination_path, self.documents_file), "w") as f:
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
    
    
    def _embed_documents(self, documents: list[str]) -> list[str]:
        """
        Embed the documents and add them to the vector database.
        
        Parameters:
        -----------
        - documents (list[str]): the list of documents paths to embed
        
        Returns:
        --------
        - list[str]: the list of embedded documents
        """
    
        # Initialize the documents mapping
        embedded_documents = []
        
        # Iterate over the files
        for document in tqdm(documents):
            # Check if the file exists
            if not os.path.exists(document):
                # Log the warning and continue
                logging.warning(f"The document {document} does not exist! Skipping...")
                continue
            
            try:
                # Opening the document in read mode
                with open(document, 'r') as d:
                    # Reading and encoding the content of the docuemnt into the embedding space
                    document_embedding = self._encode_texts([d.read()], self.documents_encoder, self.documents_tokenizer) # type: ignore
                    
            except Exception as e:
                # Log the exception and continue to the next document
                logging.error(f"Error while processing the document {document}: {e}")
                continue
                
            # Add the document embedding to the vector database
            self.vector_database.add(document_embedding) # type: ignore
                
            # Append the document path to the documents mapping
            embedded_documents.append(document)
            
        # Return the documents mapping
        return embedded_documents