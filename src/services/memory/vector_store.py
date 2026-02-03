import os
import logging
import time
from functools import lru_cache, wraps
from typing import List, Dict, Optional, Tuple

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
from openai import OpenAI, OpenAIError, RateLimitError, APITimeoutError
from dotenv import load_dotenv

from src.core.config import Settings, get_settings

# Load environment variables
load_dotenv()

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

    def __init__(self, settings: Optional[Settings] = None):
        """Initialize semantic memory store.

        Args:
            settings: Optional settings object. If not provided, loads from environment.
        """
        if settings is None:
            try:
                settings = get_settings()
            except Exception as e:
                # If settings fail to load but we aren't using them yet, we handle it below
                logger.warning(f"Could not load settings in MemoryStore: {e}")

        self.conn_string = (
            settings.memory_db_url if settings else os.getenv("MEMORY_DB_URL")
        )
        if not self.conn_string:
            raise ValueError(
                "MEMORY_DB_URL environment variable is required. "
                "Add it to your .env file with your TimescaleDB connection string."
            )

        api_key = settings.openai_api_key if settings else os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is required. "
                "Add it to your .env file."
            )

        self.openai_client = OpenAI(api_key=api_key)
        self.embedding_model = (
            settings.openai_embedding_model
            if settings
            else os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        )

        # Get embedding dimensions (auto-detect from model or use override)
        default_dim = self.EMBEDDING_DIMS.get(self.embedding_model, 1536)
        self.embedding_dim = (
            settings.openai_embedding_dim
            if settings
            else int(os.getenv("OPENAI_EMBEDDING_DIM", str(default_dim)))
        )

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
        logger.debug(
            f"OpenAI embedding request: model={self.embedding_model}, input_length={len(text)}, preview={text[:100]}..."
        )

        try:
            response = self.openai_client.embeddings.create(
                model=self.embedding_model, input=text, timeout=30.0
            )
            embedding = response.data[0].embedding

            logger.debug(
                f"OpenAI embedding response: dimensions={len(embedding)}, first_5_values={embedding[:5]}, tokens_used={response.usage.total_tokens}"
            )

            return embedding
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
        context: Optional[str] = None,
        importance: float = 1.0,
        confidence: float = 1.0,
        source: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> Optional[int]:
        """
        Store a new memory.

        Args:
            memory_text: The distilled memory bullet point
            type: Memory type (preference, fact, task, insight)
            context: Tag for organization (project name, work, personal, etc)
            importance: Importance score 0.0-3.0 (0=low, 3=high)
            confidence: Confidence score 0-1
            source: Brief snippet of what triggered this
            tag: Deprecated alias for context

        Returns:
            Memory ID on success, None on failure
        """
        if tag is not None and context is None:
            context = tag

        # Input validation
        if not memory_text or not memory_text.strip():
            raise ValueError("memory_text cannot be empty")

        if len(memory_text) > self.MAX_TEXT_LENGTH:
            raise ValueError(f"memory_text too long (max {self.MAX_TEXT_LENGTH} chars)")

        if type not in self.VALID_TYPES:
            raise ValueError(f"type must be one of {self.VALID_TYPES}")

        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

        if not 0 <= importance <= 3:
            raise ValueError("importance must be between 0 and 3")

        if not context or not context.strip():
            raise ValueError("context cannot be empty")

        conn = None
        try:
            embedding = self._get_embedding(memory_text)
            logger.debug(
                f"Generated embedding for storage: text_length={len(memory_text)}, embedding_dims={len(embedding)}, type={type}, tag={tag}"
            )

            conn = self._get_connection()

            with conn.cursor() as cur:
                cur.execute("SET ivfflat.probes = 20")

                cur.execute(
                    """
                    INSERT INTO hermes.memories
                    (memory_text, type, tag, importance, confidence, source, embedding, embedding_model, embedding_dim)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """,
                    (
                        memory_text,
                        type,
                        context,
                        importance,
                        confidence,
                        source,
                        embedding,
                        self.embedding_model,
                        self.embedding_dim,
                    ),
                )

                memory_id = cur.fetchone()[0]
                conn.commit()

                logger.info(f"Stored memory {memory_id}: {memory_text[:50]}...")
                logger.debug(
                    f"Database insert confirmed: id={memory_id}, type={type}, tag={context}, confidence={confidence}, embedding_model={self.embedding_model}"
                )

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
        min_importance: Optional[float] = None,
        limit: int = 5,
        use_semantic: bool = True,
        tag: Optional[str] = None,
    ) -> List[Dict]:
        """
        Retrieve relevant memories.

        Args:
            query: Search query
            type: Filter by memory type
            context: Filter by context (supports LIKE patterns)
            min_importance: Filter by minimum importance score
            limit: Max results to return
            use_semantic: Use semantic search (True) or full-text (False)
            tag: Deprecated alias for context

        Returns:
            List of memory dicts with similarity scores
        """
        if tag is not None and context is None:
            context = tag

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
                cur.execute("SET ivfflat.probes = 20")

                if use_semantic:
                    query_embedding = self._get_embedding(query)
                    logger.debug(
                        f"Semantic search: query_length={len(query)}, query_preview={query[:50]}..., embedding_dims={len(query_embedding)}"
                    )

                    sql = """
                        SELECT
                            id, memory_text, type, tag, importance, confidence,
                            source, created_at, last_accessed, access_count,
                            embedding_model,
                            (1 - (embedding <=> %s::vector)) * (1 + (importance / 3.0)) as similarity
                        FROM hermes.memories
                        WHERE 1=1
                    """
                    params = [query_embedding]
                else:
                    sql = """
                        SELECT
                            id, memory_text, type, tag, importance, confidence,
                            source, created_at, last_accessed, access_count,
                            embedding_model,
                            ts_rank(to_tsvector('english', memory_text), plainto_tsquery('english', %s)) * (1 + (importance / 3.0)) as similarity
                        FROM hermes.memories
                        WHERE to_tsvector('english', memory_text) @@ plainto_tsquery('english', %s)
                    """
                    params = [query, query]

                if type:
                    sql += " AND type = %s"
                    params.append(type)

                if context:
                    if "%" in context:
                        sql += " AND tag LIKE %s"
                    else:
                        sql += " AND tag = %s"
                    params.append(context)

                if min_importance is not None:
                    sql += " AND importance >= %s"
                    params.append(min_importance)

                sql += " ORDER BY similarity DESC LIMIT %s"
                params.append(limit)

                cur.execute(sql, params)
                results = cur.fetchall()

                # Update access tracking for returned memories
                if results:
                    memory_ids = [r["id"] for r in results]
                    cur.execute(
                        """
                        UPDATE hermes.memories
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
                cur.execute("DELETE FROM hermes.memories WHERE id = %s", (memory_id,))
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

    def list_memories(
        self,
        tag: Optional[str] = None,
        type: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict]:
        """List memories with optional filtering and pagination.

        Args:
            tag: Optional tag filter
            type: Optional type filter (preference, fact, task, insight)
            limit: Maximum number of results (default 20, max 100)
            offset: Number of results to skip (for pagination)

        Returns:
            List of memory dictionaries with all fields
        """
        if limit < 1 or limit > 100:
            raise ValueError("limit must be between 1 and 100")

        if offset < 0:
            raise ValueError("offset must be non-negative")

        if type and type not in self.VALID_TYPES:
            raise ValueError(f"type must be one of {self.VALID_TYPES}")

        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                sql = """
                    SELECT
                        id, memory_text, type, tag, importance, confidence,
                        source, created_at, last_accessed, access_count,
                        embedding_model
                    FROM hermes.memories
                    WHERE 1=1
                """
                params = []

                if tag:
                    sql += " AND tag = %s"
                    params.append(tag)

                if type:
                    sql += " AND type = %s"
                    params.append(type)

                sql += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
                params.extend([limit, offset])

                cur.execute(sql, params)
                results = [dict(row) for row in cur.fetchall()]
                logger.debug(
                    f"Listed {len(results)} memories (tag={tag}, type={type}, limit={limit})"
                )
                return results

        except psycopg2.Error as e:
            logger.error(f"Database error listing memories: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing memories: {e}")
            return []
        finally:
            if conn:
                self._return_connection(conn)

    def list_contexts(self) -> List[str]:
        """Get all unique contexts."""
        return self.list_tags()

    def list_tags(self) -> List[str]:
        """Get all unique tags."""
        conn = None
        try:
            conn = self._get_connection()
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT tag FROM hermes.memories ORDER BY tag")
                tags = [row[0] for row in cur.fetchall()]
                logger.debug(f"Found {len(tags)} unique tags")
                return tags
        except psycopg2.Error as e:
            logger.error(f"Database error listing tags: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing tags: {e}")
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
                # Get aggregates
                cur.execute("""
                    SELECT
                        COUNT(*) as total_memories,
                        COUNT(DISTINCT type) as total_types,
                        COUNT(DISTINCT tag) as total_tags,
                        AVG(confidence) as avg_confidence,
                        AVG(importance) as avg_importance,
                        MAX(created_at) as last_memory_at
                    FROM hermes.memories
                """)
                stats = dict(cur.fetchone())

                # Get type distribution for 'memory_types' field
                cur.execute("""
                    SELECT type, COUNT(*) as count 
                    FROM hermes.memories 
                    GROUP BY type
                """)
                type_counts = {row["type"]: row["count"] for row in cur.fetchall()}
                stats["memory_types"] = type_counts

                # Alias for backward compatibility/CLI expectations
                stats["unique_types"] = stats["total_types"]
                stats["unique_tags"] = stats["total_tags"]

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
