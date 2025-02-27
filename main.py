import logging
from src import SemanticDatabase

# Setup logging to console
logging.basicConfig(level=logging.INFO)

# Define the base directory and the local destination
base_directory = "./documents"
destination_path = "./database"

# Instantiate the semantic database
semantic_database = SemanticDatabase(base_directory)

# Creating the database
semantic_database.create_or_update(destination_path)

# Loading the database
semantic_database.load(destination_path)

# Updating the database
#semantic_database.create_or_update(destination_path)

# Executing a query
results = semantic_database.query(
    query_string = "Search for all the documents about finance",
    num_results = 5
)

# Reprojecting and plotting the embedding space in 2D
semantic_database.plot_space()

# Print the results
for result in results:
    print(result)