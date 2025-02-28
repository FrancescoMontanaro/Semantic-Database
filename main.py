import logging
from src import SemanticDatabase

# Setup logging to console
logging.basicConfig(level=logging.INFO)

# Define the base directory and the local destination
base_directory = "./documents"
destination_path = "./database"

# Instantiate the semantic database
semantic_database = SemanticDatabase(
    base_directory = base_directory,
    batch_size = 4,
    context_length = 2048
)

# Creating the database
semantic_database.create_or_update(destination_path)

# Loading the database
semantic_database.load(destination_path)

# Reprojecting and plotting the embedding space in 2D
semantic_database.plot_space()

# Executing a query
results = semantic_database.query(
    query_string = "Voglio acquistare un asset",
    num_results = 5
)

# Print the results
for result in results:
    print(result)