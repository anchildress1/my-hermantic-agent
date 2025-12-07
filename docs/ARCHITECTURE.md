# Architecture Overview

Hermes Agent architecture and data flow diagrams.

## System Architecture

```mermaid
%%{init: {'theme':'base', 'lightTheme':'base', 'darkTheme':'dark', 'securityLevel':'strict', 'accessibility': {'label': 'System Architecture', 'description': 'System architecture and data flow; supports light/dark modes.'}}}%%
graph TB
    subgraph "User Interface"
        CLI[CLI Chat Interface]
    end

    subgraph "Application Layer"
        Main[main.py]
        Chat[chat.py]
        Memory[memory.py]
    end

    subgraph "External Services"
        Ollama[Ollama<br/>Local LLM]
        OpenAI[OpenAI API<br/>Embeddings]
        TimescaleDB[(TimescaleDB<br/>Vector Store)]
    end

    subgraph "Storage"
        JSON[data/memory.json<br/>Conversations]
        Logs[logs/<br/>Application Logs]
    end

    CLI --> Main
    Main --> Chat
    Chat --> Memory
    Chat --> Ollama
    Chat --> JSON
    Chat --> Logs
    Memory --> OpenAI
    Memory --> TimescaleDB

    style CLI fill:#4A90E2
    style Ollama fill:#FF6B6B
    style OpenAI fill:#4ECDC4
    style TimescaleDB fill:#95E1D3
    style JSON fill:#F9CA24
    style Logs fill:#F9CA24
```

## Data Flow

```mermaid
%%{init: {'theme':'base', 'lightTheme':'base', 'darkTheme':'dark', 'securityLevel':'strict', 'accessibility': {'label': 'Data Flow', 'description': 'Sequence diagram showing message flow; supports light/dark modes.'}}}%%
sequenceDiagram
    participant User
    participant Chat
    participant Ollama
    participant Memory
    participant OpenAI
    participant TimescaleDB

    User->>Chat: Send message
    Chat->>Chat: Load conversation history
    Chat->>Ollama: Generate response
    Ollama-->>Chat: Stream response
    Chat-->>User: Display response
    Chat->>Chat: Save to memory.json

    alt Semantic Memory Command
        User->>Chat: /remember <text>
        Chat->>Memory: Store memory
        Memory->>OpenAI: Generate embedding
        OpenAI-->>Memory: Return vector
        Memory->>TimescaleDB: Store with embedding
        TimescaleDB-->>Memory: Confirm
        Memory-->>Chat: Success
        Chat-->>User: Memory stored
    end

    alt Recall Memory
        User->>Chat: /recall <query>
        Chat->>Memory: Search memories
        Memory->>OpenAI: Generate query embedding
        OpenAI-->>Memory: Return vector
        Memory->>TimescaleDB: Vector similarity search
        TimescaleDB-->>Memory: Return matches
        Memory-->>Chat: Relevant memories
        Chat-->>User: Display results
    end
```

## Memory System

```mermaid
%%{init: {'theme':'base', 'lightTheme':'base', 'darkTheme':'dark', 'securityLevel':'strict', 'accessibility': {'label': 'Dual Memory Architecture', 'description': 'Dual memory diagram showing short-term and long-term memory.'}}}%%
graph LR
    subgraph "Dual Memory Architecture"
        subgraph "Short-term Memory"
            Conv[Conversation History<br/>data/memory.json]
            Conv --> Trim[Auto-trim on<br/>token limit]
        end

        subgraph "Long-term Memory"
            Semantic[Semantic Memory<br/>TimescaleDB]
            Semantic --> Types[Types:<br/>preference, fact,<br/>task, insight]
            Semantic --> Context[Contexts:<br/>work, personal,<br/>project-*]
        end
    end

    User[User Input] --> Conv
    Conv --> LLM[Ollama LLM]
    LLM --> Response[Response]

    User --> |/remember| Semantic
    Semantic --> |/recall| Response

    style Conv fill:#FFE66D
    style Semantic fill:#95E1D3
    style LLM fill:#FF6B6B
```

## Component Responsibilities

```mermaid
%%{init: {'theme':'base', 'lightTheme':'base', 'darkTheme':'dark', 'securityLevel':'strict', 'accessibility': {'label': 'Component Responsibilities', 'description': 'Diagram describing component responsibilities across modules.'}}}%%
graph TD
    subgraph "main.py"
        A1[Environment Validation]
        A2[Logging Setup]
        A3[Template Loading]
        A4[Chat Loop Initialization]
    end

    subgraph "chat.py"
        B1[Ollama Integration]
        B2[Conversation Management]
        B3[Context Trimming]
        B4[File I/O - Atomic Writes]
        B5[Command Processing]
        B6[Memory Integration]
    end

    subgraph "memory.py"
        C1[Connection Pooling]
        C2[Embedding Generation]
        C3[Vector Search]
        C4[CRUD Operations]
        C5[Error Handling]
        C6[Rate Limiting]
    end

    A1 --> A2 --> A3 --> A4
    A4 --> B1
    B1 --> B2 --> B3 --> B4 --> B5 --> B6
    B6 --> C1
    C1 --> C2 --> C3 --> C4 --> C5 --> C6

    style A1 fill:#4A90E2
    style B1 fill:#FF6B6B
    style C1 fill:#95E1D3
```

## Error Handling Flow

```mermaid
%%{init: {'theme':'base', 'lightTheme':'base', 'darkTheme':'dark', 'securityLevel':'strict', 'accessibility': {'label': 'Error Handling Flow', 'description': 'Error handling and retry flow; supports light/dark modes.'}}}%%
graph TD
    Start[Operation Start] --> Try{Try Operation}
    Try -->|Success| Log1[Log Success]
    Try -->|Database Error| DB[Handle DB Error]
    Try -->|API Error| API[Handle API Error]
    Try -->|Validation Error| Val[Handle Validation]

    DB --> Retry{Retry?}
    API --> Retry
    Retry -->|Yes| Try
    Retry -->|No| Log2[Log Error]

    Val --> Log2
    Log1 --> Return[Return Result]
    Log2 --> Return

    style Try fill:#4A90E2
    style DB fill:#FF6B6B
    style API fill:#FF6B6B
    style Val fill:#FF6B6B
    style Log1 fill:#95E1D3
    style Log2 fill:#F9CA24
```

## Technology Stack

```mermaid
%%{init: {'theme':'base', 'lightTheme':'base', 'darkTheme':'dark', 'securityLevel':'strict', 'accessibility': {'label': 'Technology Stack', 'description': 'Tech stack components and their relationships; supports light/dark modes.'}}}%%
graph LR
    subgraph "Frontend"
        CLI[CLI Interface<br/>Python]
    end

    subgraph "Backend"
        App[Application<br/>Python 3.12+]
        UV[Package Manager<br/>uv]
    end

    subgraph "AI/ML"
        OL[Ollama<br/>Local LLM]
        OAI[OpenAI<br/>Embeddings API]
    end

    subgraph "Data"
        TS[(TimescaleDB<br/>pgvector)]
        FS[File System<br/>JSON]
    end

    CLI --> App
    App --> UV
    App --> OL
    App --> OAI
    App --> TS
    App --> FS

    style CLI fill:#4A90E2
    style App fill:#4A90E2
    style OL fill:#FF6B6B
    style OAI fill:#4ECDC4
    style TS fill:#95E1D3
    style FS fill:#F9CA24
```
