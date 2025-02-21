from src import SemanticDatabase

# Instantiate the semantic database
semantic_database = SemanticDatabase()

# Define the paths of the database and the local destination
documents_path = "YOUR_PATH_TO_THE_DOCUMENTS"
destination_path = "YOUR_PATH_TO_THE_DESTINATION"

# Define the paths of the query and documents encoders
query_encoder_id = "facebook/dpr-question_encoder-single-nq-base"
documents_encoder_id = "facebook/dpr-ctx_encoder-single-nq-base"

# Creating the database
semantic_database.create(
    documents_path = documents_path,
    query_encoder_id = query_encoder_id,
    documents_encoder_id = documents_encoder_id,
    destination_path = destination_path
)

# Loading the database
semantic_database.load(destination_path)

# Update the database with new documents
semantic_database.update(
    documents_path = "/Users/francescomontanaro/Desktop/Test",
    destination_path = destination_path
)

# Executing a query
results = semantic_database.query(
    query_string = "Search for all the documents about finance",
    num_results = 5
)

# Print the results
for result in results:
    print(result)