#!/usr/bin/env python3
"""Helper script to retrieve database password from Tiger Cloud."""

import sys
import os
from dotenv import load_dotenv

load_dotenv()


def main():
    """Get database connection string with password."""
    
    # Check if we have the service ID from the connection string
    db_url = os.getenv("MEMORY_DB_URL")
    if not db_url:
        print("❌ MEMORY_DB_URL not found in .env")
        return 1
    
    # Extract service ID from hostname (format: <service_id>.*.tsdb.cloud.timescale.com)
    try:
        # Parse hostname from connection string
        # postgresql://user@host:port/db
        parts = db_url.split("@")[1].split(":")[0]  # Get host part
        service_id = parts.split(".")[0]  # Get first part before first dot
        
        print(f"📋 Service ID: {service_id}")
        print(f"\n🔑 To get your password, use the Tiger Cloud MCP tools:")
        print(f"\n   Ask me: 'Get the password for service {service_id}'")
        print(f"\n   Or use the Tiger Cloud console: https://console.timescale.cloud")
        print(f"\n   Then update your .env file with the full connection string including password:")
        print(f"   MEMORY_DB_URL=postgresql://tsdbadmin:PASSWORD@{parts}:33500/tsdb?sslmode=require")
        
    except Exception as e:
        print(f"❌ Could not parse service ID from connection string: {e}")
        print(f"\nYour connection string: {db_url}")
        print(f"\nTo get your password:")
        print(f"1. List your services to find the service ID")
        print(f"2. Get service details with password")
        print(f"3. Update MEMORY_DB_URL in .env with the password")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
