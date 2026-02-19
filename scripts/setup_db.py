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
    app_password = os.getenv("HERMES_APP_PASSWORD", "hermes_app_password")

    if not db_url:
        print("❌ MEMORY_DB_URL not set in .env file")
        print("\nAdd your TimescaleDB connection string to .env:")
        print(
            "  MEMORY_DB_URL=postgresql://user:password@host:port/database?sslmode=require"
        )
        return 1

    schema_dir = Path("schema")
    migration_files = sorted(schema_dir.glob("*.sql"))
    if not migration_files:
        print(f"❌ No schema files found in: {schema_dir}")
        return 1

    print("🔌 Connecting to database...")
    try:
        conn = psycopg2.connect(db_url)
        print("✓ Connected successfully")

        with conn.cursor() as cur:
            for migration_file in migration_files:
                print(f"📝 Applying migration: {migration_file}")
                with open(migration_file, "r") as f:
                    migration_sql = f.read()

                # Replace placeholder password with env var
                migration_sql = migration_sql.replace(
                    "hermes_app_password_placeholder", app_password
                )
                cur.execute(migration_sql)
                conn.commit()

            print(f"✓ Applied {len(migration_files)} migration file(s)")

            # Verify tables exist
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'hermes'
                AND table_name = 'memories'
            """)

            if cur.fetchone():
                print("✓ memories table created")

                # Check for indexes
                cur.execute("""
                    SELECT indexname
                    FROM pg_indexes
                    WHERE tablename = 'memories'
                """)
                indexes = [row[0] for row in cur.fetchall()]
                print(f"✓ Created {len(indexes)} indexes: {', '.join(indexes)}")
            else:
                print("⚠️  Warning: memories table not found")

            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'hermes'
                AND table_name = 'memory_events'
            """
            )
            if cur.fetchone():
                print("✓ memory_events table created")
            else:
                print("⚠️  Warning: memory_events table not found")

        conn.close()
        print("\n✅ Database setup complete!")
        print("\nTest the connection:")
        print(
            '  uv run python -c "from src.services.memory.vector_store import MemoryStore; print(MemoryStore().stats())"'
        )
        return 0

    except psycopg2.Error as e:
        print(f"\n❌ Database error: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
