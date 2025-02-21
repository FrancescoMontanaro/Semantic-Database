import os
    
     
def list_files(path: str, extensions: list[str] = []) -> list[str]:
    """
    List the files in a directory with the specified extensions.
    
    Parameters:
    ----------
    - path (str): the path of the directory from which to list the files
    - extensions: (list[str]): the list of extensions to filter the files
    
    Returns:
    --------
    - list[str]: the list of files in the directory
    
    Raises:
    -------
    - Exception: if the specified path do not exists
    """
    
    # Check if the path exists
    if not os.path.exists(path):
        raise Exception(f"The specified path does not exist: {path}")
    
    # Create a list to store the files with the specified extensions
    files = []
    
    # Walk through the directory and subdirectories
    for root, _, filenames in os.walk(path):
        for file in filenames:
            # If no extensions are specified, add all files
            # Otherwise, add only the files with the specified extensions
            if len(extensions) == 0 or any([file.endswith(ext) for ext in extensions]):
                # Append the file to the list
                files.append(os.path.join(root, file))
    
    # Return the list of files
    return files