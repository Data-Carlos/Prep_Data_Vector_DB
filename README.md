# Tutorial: Preparing arXiv Articles for Embedding Models

In this tutorial, we'll guide you through the process of downloading, processing, and chunking a dataset of arXiv papers. This tutorial is essential for anyone looking to manage and utilize large text datasets, especially for applications involving machine learning and embedding models.

## Objective

By the end of this tutorial, you will be able to:
- Download the arXiv metadata dataset.
- Extract relevant metadata based on specific criteria.
- Download PDFs of selected papers.
- Convert PDFs to text files.
- Chunk text files and add metadata to each chunk.
- Prepare the dataset for use in an embedding model by converting all data to strings.
- Save the processed data to a CSV file.

## Structure

The tutorial is broken down into the following steps:
1. Download the Metadata Dataset from Kaggle.
2. Extract Relevant Metadata.
3. Download PDFs of the Papers.
4. Convert PDFs to Text.
5. Chunk Text and Add Metadata.
6. Convert Data to Strings for Embedding Model.
7. Save the DataFrame to a CSV File.
8. Conclusion.

## Audience

This tutorial is intended for:
- Intermediate users familiar with Python programming.
- Individuals with basic knowledge of data processing and machine learning.
- Researchers or data scientists looking to work with large text datasets.

## Resources

You will need the following:
- Kaggle account and API key.
- Python environment with necessary libraries (e.g., `json`, `datetime`, `requests`, `os`, `PyPDF2`, `pandas`).

## Steps

### Step 1: Download the Metadata Dataset from Kaggle

First, you need to download the arXiv metadata dataset from Kaggle. You can use the Kaggle API or a direct download link. To download the dataset using `curl`, run the following command in your terminal:

```bash
curl -L -o arxiv-metadata-oai-snapshot.json https://www.kaggle.com/datasets/Cornell-University/arxiv
```

### Step 2: Extract Relevant Metadata

We will process the metadata to extract the IDs of papers updated after a certain date.

```python
import json
from datetime import datetime

# Define the date cutoff
date_cutoff = datetime.strptime("2024-04-01", "%Y-%m-%d")

# Path to the JSON metadata file
metadata_path = "arxiv-metadata-oai-snapshot.json"

# Extract relevant metadata
paper_ids = []
with open(metadata_path, encoding="utf8") as f:
    for entry in f:
        data = json.loads(entry)
        update_date = datetime.strptime(data.get("update_date", ""), "%Y-%m-%d")
        title = data.get("title", "")
        if update_date >= date_cutoff and "GPT-4 Vision" in title:
            paper_ids.append(data.get("id", ""))

print(f"Found {len(paper_ids)} papers with 'GPT-4 Vision' in the title.")
```

### Step 3: Download PDFs of the Papers

Next, we will download the PDFs of the selected papers.

```python
import os
import requests

# Construct the list of PDF URLs
base_url = "https://arxiv.org/pdf/"
pdf_urls = [f"{base_url}{paper_id}.pdf" for paper_id in paper_ids]

# Ensure the directory to save downloaded PDFs exists
download_directory = "./downloaded_pdfs/"
os.makedirs(download_directory, exist_ok=True)

# Function to download a single PDF
def download_pdf(url, directory):
    response = requests.get(url)
    if response.status_code == 200:
        filename = os.path.join(directory, url.split('/')[-1])
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {url}")

# Download all PDFs
for pdf_url in pdf_urls:
    download_pdf(pdf_url, download_directory)

print("Download complete.")
```

### Step 4: Convert PDFs to Text

We will convert the downloaded PDFs to text files for easier processing.

```python
from PyPDF2 import PdfReader

# Define the directory containing the downloaded PDFs
pdf_directory = "./downloaded_pdfs/"

# Define the directory to save the TXT files
txt_directory = "./converted_txt/"
os.makedirs(txt_directory, exist_ok=True)

# Function to convert a single PDF to TXT
def convert_pdf_to_txt(pdf_path, txt_path):
    try:
        with open(pdf_path, 'rb') as pdffileobj:
            pdfreader = PdfReader(pdffileobj)
            text = ""
            for page in pdfreader.pages:
                text += page.extract_text()
            
            with open(txt_path, 'w', encoding='utf-8') as txtfile:
                txtfile.write(text)
            
            print(f"Converted {pdf_path} to {txt_path}")
    except Exception as e:
        print(f"Failed to convert {pdf_path}: {e}")

# Iterate over all PDFs in the directory and convert them to TXT
for pdf_filename in os.listdir(pdf_directory):
    if pdf_filename.endswith('.pdf'):
        pdf_path = os.path.join(pdf_directory, pdf_filename)
        txt_filename = os.path.splitext(pdf_filename)[0] + '.txt'
        txt_path = os.path.join(txt_directory, txt_filename)
        convert_pdf_to_txt(pdf_path, txt_path)
```

### Step 5: Chunk Text and Add Metadata

We will chunk the text files into smaller parts and add metadata for each chunk.

```python
import pandas as pd

# Path to the JSON metadata file
metadata_path = "arxiv-metadata-oai-snapshot.json"

# Extract metadata and store in a dictionary
metadata_dict = {}
with open(metadata_path, encoding="utf8") as f:
    for entry in f:
        data = json.loads(entry)
        paper_id = data.get("id", "")
        metadata_dict[paper_id] = data

# Function to chunk text
def chunk_text(text, chunk_size=1000):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

# Define the directory containing the TXT files
txt_directory = "./converted_txt/"

# Store the chunks and their metadata
chunks_data = []

for txt_filename in os.listdir(txt_directory):
    if txt_filename.endswith('.txt'):
        paper_id = os.path.splitext(txt_filename)[0]
        txt_path = os.path.join(txt_directory, txt_filename)
        
        # Read the TXT file contents
        with open(txt_path, 'r', encoding='utf-8') as txtfile:
            text = txtfile.read()
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Get the metadata for this paper
        if paper_id in metadata_dict:
            metadata = metadata_dict[paper_id]
        else:
            print(f"Metadata not found for paper ID: {paper_id}")
            continue
        
        # Store each chunk with its metadata
        for i, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_{i}"
            chunk_data = {
                "chunk_id": chunk_id,
                "chunk": chunk,
            }
            # Add metadata fields to chunk_data
            chunk_data.update(metadata)
            chunks_data.append(chunk_data)

# Convert the data to a DataFrame
df = pd.DataFrame(chunks_data)

# Define the path to the output CSV file
csv_output_path = "chunks_metadata.csv"

# Save the DataFrame to a CSV file
df.to_csv(csv_output_path, index=False)

print("CSV file created successfully.")
```

### Step 6: Convert Data to Strings for Embedding Model

To prepare the dataset for an embedding model, we need to ensure all data is converted to strings.

```python
# Convert all data to strings
for chunk in chunks_data:
    for key, value in chunk.items():
        chunk[key] = str(value)
```

### Step 7: Save the DataFrame to a CSV File

Finally, we will save the DataFrame to a CSV file.

```python
# Convert the data to a DataFrame
df = pd.DataFrame(chunks_data)

# Define the path to the output CSV file
csv_output_path = "chunks_metadata.csv"

# Save the DataFrame to a CSV file
df.to_csv(csv_output_path, index=False)

print("CSV file created successfully.")
```

## Conclusion

In this tutorial, we covered how to download, process, and chunk a dataset of arXiv papers. We also added a step to convert all data to strings, preparing it for use in an embedding model. By following these steps, you can prepare a chunked dataset ready for use in a vector database. This process transforms you from a beginner to a proficient user capable of managing and utilizing large text datasets.

Feel free to ask questions or provide feedback to improve this tutorial further. Happy coding!
