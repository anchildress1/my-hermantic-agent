#!/usr/bin/env python3
"""Setup script for initializing the TimescaleDB database."""

import os
import sys
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def main():
    """Initialize the database schema."""
    db_url = os.getenv("MEMORY_DB_URL")
    
    if not db_url:
        print("❌ MEMORY_DB_URL not set in .env file")
        print("\nAdd your TimescaleDB connection string to .env:")
        print("  MEMORY_DB_URL=postgresql://user:password@host:port/database?sslmode=require")
        return 1
    
    schema_file = Path("schema/init.sql")
    if not schema_file.exists():
        print(f"❌ Schema file not found: {schema_file}")
        return 1
    
    print(f"📋 Reading schema from {schema_file}")
    with open(schema_file, "r") as f:
        schema_sql = f.read()
    
    print(f"🔌 Connecting to database...")
    try:
        conn = psycopg2.connect(db_url)
        print("✓ Connected successfully")
        
        with conn.cursor() as cur:
            print("📝 Executing schema...")
            cur.execute(schema_sql)
            conn.commit()
            print("✓ Schema initialized successfully")
            
            # Verify tables exist
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public' 
                AND table_name = 'agent_memories'
            """)
            
            if cur.fetchone():
                print("✓ agent_memories table created")
                
                # Check for indexes
                cur.execute("""
                    SELECT indexname 
                    FROM pg_indexes 
                    WHERE tablename = 'agent_memories'
                """)
                indexes = [row[0] for row in cur.fetchall()]
                print(f"✓ Created {len(indexes)} indexes: {', '.join(indexes)}")
            else:
                print("⚠️  Warning: agent_memories table not found")
        
        conn.close()
        print("\n✅ Database setup complete!")
        print("\nTest the connection:")
        print("  uv run python -c \"from src.agent.memory import MemoryStore; print(MemoryStore().stats())\"")
        return 0
        
    except psycopg2.Error as e:
        print(f"\n❌ Database error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
