#!/bin/bash

# Directory containing the books
BOOK_DIR="./book"

# Loop through all .txt files in the directory
for book_file in "$BOOK_DIR"/*.txt; do
    # Extract the book name (prefix before .txt)
    book_name=$(basename "$book_file" .txt)
    
    # Execute rel.py with the book name
    echo "Processing: $book_name"
    python rel.py --book "$book_name"
done