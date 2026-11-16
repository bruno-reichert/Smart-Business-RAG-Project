import sys
from pathlib import Path

# Add project root to sys.path so 'app' is always discoverable
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from dotenv import load_dotenv
from langchain_core.documents import Document

from app.models.schemas import Citation, OperationalInsight

# Ensure .env is loaded before any tests run
load_dotenv()

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Returns the absolute Path to the repository root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """Returns the absolute Path to the data directory containing PDF filings."""
    return project_root / "data"


@pytest.fixture
def sample_document() -> Document:
    """Provides a sample LangChain Document representing a page from a quarterly filing."""
    return Document(
        page_content=(
            "Nubank delivered strong results in Brazil, reaching 110 million customers "
            "with an activity rate exceeding 85%. Meanwhile, deposits in Mexico grew "
            "substantially to over $3.0 billion."
        ),
        metadata={
            "source": "nubank-3q25.pdf",
            "company": "Nubank",
            "fiscal_period": "3Q25",
            "page": 1,
            "total_pages": 4,
        },
    )


@pytest.fixture
def sample_citation() -> Citation:
    """Provides a valid Citation instance."""
    return Citation(
        document="nubank-3q25.pdf",
        page=1,
        quote="reaching 110 million customers with an activity rate exceeding 85%",
    )


@pytest.fixture
def sample_insight(sample_citation: Citation) -> OperationalInsight:
    """Provides a valid structured OperationalInsight instance."""
    return OperationalInsight(
        answer="Nubank reached 110 million customers in Brazil with an activity rate over 85%.",
        citations=[sample_citation],
        is_grounded=True,
    )