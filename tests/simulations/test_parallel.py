import concurrent.futures
import os

def process_item(item):
    # Your processing logic here
    result = item * 2
    
    # Example: Creating a file
    filename = f"result_{item}.txt"
    with open(filename, "w") as file:
        file.write(str(result))

    # No explicit return needed

def parallel_process(items):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # The map function will return None for each item
        executor.map(process_item, items)

if __name__ == "__main__":
    # Example list of items to process
    items_to_process = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Parallelize the processing of items
    parallel_process(items_to_process)

    print("Processing complete. Check for created files.")
