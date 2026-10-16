import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def test_database() -> None:
    # 1. Guard clause: Ensure DATABASE_URL is not None
    if not DATABASE_URL:
        print("Error: DATABASE_URL is not set in your .env file!")
        sys.exit(1)

    print("Testing connection to Supabase PostgreSQL...")
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            # Check Postgres version
            res = conn.execute(text("SELECT version();")).fetchone()
            
            # 2. Guard clause: Ensure res is not None before subscripting
            if res is not None:
                version_str = str(res[0])
                print(f"Connected! Postgres Version: {version_str[:40]}...")
            else:
                print("Connected, but could not read database version.")
            
            # Check if pgvector is enabled
            vector_check = conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector';")
            ).fetchone()
            
            if vector_check is not None:
                print("pgvector extension is ACTIVE and ready to store embeddings!")
            else:
                print("Warning: Connected, but 'vector' extension is NOT active yet.")
                print("Run: 'CREATE EXTENSION IF NOT EXISTS vector;' in Supabase SQL editor.")
                
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_database()