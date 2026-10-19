import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Path safety
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langsmith import Client

load_dotenv()

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
if not LANGSMITH_API_KEY:
    raise ValueError("LANGSMITH_API_KEY is missing from .env!")

# Expanded Golden Dataset with business Q&A and target citations
EXPANDED_GOLDEN_DATASET = [
    {
        "inputs": {"query": "What is Nubank's customer base size and activity rate in Brazil?"},
        "outputs": {
            "expected_page": 1,
            "expected_quarter": "3Q25",
            "expected_facts": "110 million customers, activity rate above 85%"
        }
    },
    {
        "inputs": {"query": "What are the customer growth milestones in Mexico and Colombia?"},
        "outputs": {
            "expected_page": 1,
            "expected_quarter": "3Q25",
            "expected_facts": "Mexico surpassed 13 million; Colombia approaching 4 million"
        }
    },
    {
        "inputs": {"query": "What was Nubank's total credit portfolio size and year-over-year growth rate?"},
        "outputs": {
            "expected_page": 1,
            "expected_quarter": "3Q25",
            "expected_facts": "Credit portfolio reached $30.4 billion, up 42% YoY FXN"
        }
    },
    {
        "inputs": {"query": "What were the business highlights and new product launches across markets?"},
        "outputs": {
            "expected_page": 3,
            "expected_quarter": "4Q25",
            "expected_facts": "Over 100 new products, payroll loans in Brazil"
        }
    },
    {
        "inputs": {"query": "How is Nubank positioned relative to private financial institutions in Brazil?"},
        "outputs": {
            "expected_page": 1,
            "expected_quarter": "4Q25",
            "expected_facts": "Largest private financial institution by number of customers (Central Bank)"
        }
    },
    {
        "inputs": {"query": "What was the growth in secured and unsecured lending in 3Q25?"},
        "outputs": {
            "expected_page": 1,
            "expected_quarter": "3Q25",
            "expected_facts": "Secured grew 133%, unsecured grew 63%"
        }
    }
]

def main():
    print("Connecting to LangSmith Client...")
    client = Client()
    
    dataset_name = "nubank-operational-qa"
    description = "Golden dataset of executive operational and financial queries against Nubank quarterly filings."

    # Check if dataset already exists, or create a fresh one
    if client.has_dataset(dataset_name=dataset_name):
        print(f"Dataset '{dataset_name}' already exists. Deleting older version for a clean upload...")
        client.delete_dataset(dataset_name=dataset_name)

    print(f"Creating fresh LangSmith dataset: '{dataset_name}'...")
    dataset = client.create_dataset(
        dataset_name=dataset_name,
        description=description
    )

    # Upload examples
    print(f"Uploading {len(EXPANDED_GOLDEN_DATASET)} golden examples to LangSmith...")
    for item in EXPANDED_GOLDEN_DATASET:
        client.create_example(
            inputs=item["inputs"],
            outputs=item["outputs"],
            dataset_id=dataset.id
        )

    print(f"\n🎉 Successfully uploaded {len(EXPANDED_GOLDEN_DATASET)} examples to LangSmith!")
    print(f"View it live at: https://smith.langchain.com")

if __name__ == "__main__":
    main()