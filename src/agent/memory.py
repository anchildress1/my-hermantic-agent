import os
import logging
import time
from functools import lru_cache, wraps
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from openai import OpenAI, OpenAIError, RateLimitError, APITimeoutError
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logger = logging.getLogger(__name__)


def rate_limit(max_calls: int, period: float):
    """Decorator to rate limit method calls."""
    calls = []

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time.time()
            # Remove old calls outside period
            calls[:] = [c for c in calls if c > now - period]

            if len(calls) >= max_calls:
                wait = period - (now - calls[0])
                raise Exception(f"Rate limit exceeded. Wait {wait:.1f}s")

            calls.append(now)
            return func(*args, **kwargs)

        return wrapper

    return decorator


class MemoryStore:
    """Agent memory storage with semantic search."""

    # Valid memory types
    VALID_TYPES = {"preference", "fact", "task", "insight"}

    # OpenAI embedding limits
    MAX_TEXT_LENGTH = 8000

    # Default embedding dimensions by model
    EMBEDDING_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self):
        self.conn_string = os.getenv("MEMORY_DB_URL")
        if not self.conn_string:
            raise ValueError(
                "MEMORY_DB_URL environment variable is required. "
                "Add it to your .env file with your TimescaleDB connection string."
            )

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Add it to your .env file."
            )

        self.openai_client = OpenAI(api_key=api_key)
        self.embedding_model = os.getenv(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )

        # Get embedding dimensions (auto-detect from model or use override)
        default_dim = self.EMBEDDING_DIMS.get(self.embedding_model, 1536)
        self.embedding_dim = int(os.getenv("OPENAI_EMBEDDING_DIM", str(default_dim)))

        logger.info(
            f"Using embedding model: {self.embedding_model} ({self.embedding_dim} dimensions)"
        )

        # Initialize connection pool
        try:
            self.conn_pool = pool.SimpleConnectionPool(
                minconn=1, maxconn=5, dsn=self.conn_string
            )
            logger.info("Database connection pool initialized")
        except psycopg2.Error as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise

    def _get_connection(self):
        """Get database connection from pool."""
        try:
            conn = self.conn_pool.getconn()
            if conn:
                return conn
            raise Exception("Failed to get connection from pool")
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _return_connection(self, conn):
        """Return connection to pool."""
        if conn:
            self.conn_pool.putconn(conn)

    @lru_cache(maxsize=100)
    def _get_embedding_cached(self, text: str) -> Tuple[float, ...]:
        """Generate embedding with caching (tuple for hashability)."""
        return tuple(self._get_embedding(text))

    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI with error handling."""
        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model, input=text, timeout=30.0
            )
            return response.data[0].embedding
        except RateLimitError as e:
            logger.error(f"OpenAI rate limit exceeded: {e}")
            raise Exception("OpenAI rate limit exceeded. Please try again later.")
        except APITimeoutError as e:
            logger.error(f"OpenAI API timeout: {e}")
            raise Exception("Embedding generation timed out")
        except OpenAIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"Failed to generate embedding: {e}")
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {e}")
            raise

    @rate_limit(max_calls=10, period=60.0)
    def remember(
        self,
        memory_text: str,
        type: str,
        context: str,
        confidence: float = 1.0,
        source_context: Optional[str] = None,
    ) -> Optional[int]:
        """
        Store a new memory.

        Args:
            memory_text: The distilled memory bullet point
            type: Memory type (preference, fact, task, insight)
            context: Context tag (project name, work, personal, etc)
            confidence: Confidence score 0-1
            source_context: Brief snippet of what triggered this

        Returns:
            Memory ID on success, None on failure
        """
        # Input validation
        if not memory_text or not memory_text.strip():
            raise ValueError("memory_text cannot be empty")

        if len(memory_text) > self.MAX_TEXT_LENGTH:
            raise ValueError(f"memory_text too long (max {self.MAX_TEXT_LENGTH} chars)")

        if type not in self.VALID_TYPES:
            raise ValueError(f"type must be one of {self.VALID_TYPES}")

        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

        if not context or not context.strip():
            raise ValueError("context cannot be empty")

        conn = None
        try:
            embedding = self._get_embedding(memory_text)
            conn = self._get_connection()

            with conn.cursor() as cur:
                # Set ivfflat probes for better accuracy
                cur.execute("SET ivfflat.probes = 20")

                cur.execute(
                    """
                    INSERT INTO agent_memories
                    (memory_text, type, context, confidence, source_context, embedding, embedding_model)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (
                        memory_text,
                        type,
                        context,
                        confidence,
                        source_context,
                        embedding,
                        self.embedding_model,
                    ),
                )

                memory_id = cur.fetchone()[0]
                conn.commit()
                logger.info(f"Stored memory {memory_id}: {memory_text[:50]}...")
                return memory_id

        except psycopg2.OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            return None
        except psycopg2.Error as e:
            logger.error(f"Database error storing memory: {e}")
            if conn:
                conn.rollback()
            return None
        except Exception as e:
            logger.error(f"Unexpected error storing memory: {e}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                self._return_connection(conn)

    @rate_limit(max_calls=20, period=60.0)
    def recall(
        self,
        query: str,
        type: Optional[str] = None,
        context: Optional[str] = None,
        limit: int = 5,
        use_semantic: bool = True,
    ) -> List[Dict]:
        """
        Retrieve relevant memories.

        Args:
            query: Search query
            type: Filter by memory type
            context: Filter by context (supports LIKE patterns)
            limit: Max results to return
            use_semantic: Use semantic search (True) or full-text (False)

        Returns:
            List of memory dicts with similarity scores
        """
        # Input validation
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        if type and type not in self.VALID_TYPES:
            raise ValueError(f"type must be one of {self.VALID_TYPES}")

        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")

        conn = None
        try:
            conn = self._get_connection()

            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                # Set ivfflat probes
                cur.execute("SET ivfflat.probes = 20")

                if use_semantic:
                    query_embedding = self._get_embedding(query)

                    sql = """
                        SELECT
                            id, memory_text, type, context, confidence,
                            source_context, created_at, last_accessed, access_count,
                            embedding_model,
                            1 - (embedding <=> %s::vector) as similarity
                        FROM agent_memories
                        WHERE 1=1
                    """
                    params = [query_embedding]
                else:
                    sql = """
                        SELECT
                            id, memory_text, type, context, confidence,
                            source_context, created_at, last_accessed, access_count,
                            embedding_model,
                            ts_rank(to_tsvector('english', memory_text), plainto_tsquery('english', %s)) as similarity
                        FROM agent_memories
                        WHERE to_tsvector('english', memory_text) @@ plainto_tsquery('english', %s)
                    """
                    params = [query, query]

                if type:
                    sql += " AND type = %s"
                    params.append(type)

                if context:
                    if "%" in context:
                        sql += " AND context LIKE %s"
                    else:
                        sql += " AND context = %s"
                    params.append(context)

                sql += " ORDER BY similarity DESC LIMIT %s"
                params.append(limit)

                cur.execute(sql, params)
                results = cur.fetchall()

                # Update access tracking for returned memories
                if results:
                    memory_ids = [r["id"] for r in results]
                    cur.execute(
                        """
                        UPDATE agent_memories
                        SET last_accessed = NOW(), access_count = access_count + 1
                        WHERE id = ANY(%s)
                    """,
                        (memory_ids,),
                    )
                    conn.commit()

                logger.info(
                    f"Recalled {len(results)} memories for query: {query[:50]}..."
                )
                return [dict(r) for r in results]

        except psycopg2.OperationalError as e:
            logger.error(f"Database connection failed: {e}")
            return []
        except psycopg2.Error as e:
            logger.error(f"Database error recalling memories: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error recalling memories: {e}")
            return []
        finally:
            if conn:
                self._return_connection(conn)

    def forget(self, memory_id: int) -> bool:
        """Delete a memory by ID."""
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("DELETE FROM agent_memories WHERE id = %s", (memory_id,))
                conn.commit()
                deleted = cur.rowcount > 0
                if deleted:
                    logger.info(f"Deleted memory {memory_id}")
                else:
                    logger.warning(f"Memory {memory_id} not found")
                return deleted
        except psycopg2.Error as e:
            logger.error(f"Database error deleting memory: {e}")
            if conn:
                conn.rollback()
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting memory: {e}")
            if conn:
                conn.rollback()
            return False
        finally:
            if conn:
                self._return_connection(conn)

    def list_contexts(self) -> List[str]:
        """Get all unique contexts."""
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT DISTINCT context FROM agent_memories ORDER BY context"
                )
                contexts = [row[0] for row in cur.fetchall()]
                logger.debug(f"Found {len(contexts)} unique contexts")
                return contexts
        except psycopg2.Error as e:
            logger.error(f"Database error listing contexts: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing contexts: {e}")
            return []
        finally:
            if conn:
                self._return_connection(conn)

    def stats(self) -> Optional[Dict]:
        """Get memory statistics."""
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        COUNT(*) as total_memories,
                        COUNT(DISTINCT type) as unique_types,
                        COUNT(DISTINCT context) as unique_contexts,
                        AVG(confidence) as avg_confidence,
                        MAX(created_at) as last_memory_at
                    FROM agent_memories
                """)
                stats = dict(cur.fetchone())
                logger.debug(f"Memory stats: {stats}")
                return stats
        except psycopg2.Error as e:
            logger.error(f"Database error getting stats: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error getting stats: {e}")
            return None
        finally:
            if conn:
                self._return_connection(conn)

    def close(self):
        """Close all connections in the pool."""
        if hasattr(self, "conn_pool") and self.conn_pool:
            self.conn_pool.closeall()
            logger.info("Connection pool closed")
