import os
import psycopg2
from psycopg2.extras import RealDictCursor
from openai import OpenAI
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class MemoryStore:
    """Agent memory storage with semantic search."""
    
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
        self.embedding_model = "text-embedding-3-small"
    
    def _get_connection(self):
        """Get database connection."""
        return psycopg2.connect(self.conn_string)
    
    def _get_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using OpenAI."""
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def remember(
        self,
        memory_text: str,
        type: str,
        context: str,
        confidence: float = 1.0,
        source_context: Optional[str] = None
    ) -> int:
        """
        Store a new memory.
        
        Args:
            memory_text: The distilled memory bullet point
            type: Memory type (preference, fact, task, insight)
            context: Context tag (project name, work, personal, etc)
            confidence: Confidence score 0-1
            source_context: Brief snippet of what triggered this
        
        Returns:
            Memory ID
        """
        embedding = self._get_embedding(memory_text)
        
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Set ivfflat probes for better accuracy
                cur.execute("SET ivfflat.probes = 20")
                
                cur.execute("""
                    INSERT INTO agent_memories 
                    (memory_text, type, context, confidence, source_context, embedding, embedding_model)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (memory_text, type, context, confidence, source_context, embedding, self.embedding_model))
                
                memory_id = cur.fetchone()[0]
                conn.commit()
                return memory_id
    
    def recall(
        self,
        query: str,
        type: Optional[str] = None,
        context: Optional[str] = None,
        limit: int = 5,
        use_semantic: bool = True
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
        with self._get_connection() as conn:
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
                    if '%' in context:
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
                    memory_ids = [r['id'] for r in results]
                    cur.execute("""
                        UPDATE agent_memories 
                        SET last_accessed = NOW(), access_count = access_count + 1
                        WHERE id = ANY(%s)
                    """, (memory_ids,))
                    conn.commit()
                
                return [dict(r) for r in results]
    
    def forget(self, memory_id: int) -> bool:
        """Delete a memory by ID."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM agent_memories WHERE id = %s", (memory_id,))
                conn.commit()
                return cur.rowcount > 0
    
    def list_contexts(self) -> List[str]:
        """Get all unique contexts."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT context FROM agent_memories ORDER BY context")
                return [row[0] for row in cur.fetchall()]
    
    def stats(self) -> Dict:
        """Get memory statistics."""
        with self._get_connection() as conn:
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
                return dict(cur.fetchone())
