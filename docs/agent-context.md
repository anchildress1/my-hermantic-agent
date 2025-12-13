# Context Commands

Chat history management commands for local conversation state.

## Terminology

**CONTEXT**: Local JSON file (`data/memory.json`) storing conversation history for session continuity.

## Commands

### /save

Save current conversation to disk.

**Syntax:**

```bash
/save
```

**Behavior:**

- Writes messages to `data/memory.json`
- Creates atomic backup at `data/memory.json.bak`
- Auto-saves on `/bye` or Ctrl+C

### /load

Load saved conversation from disk.

**Syntax:**

```bash
/load [file1] [file2] ...
```

**Behavior:**

- No argument: Loads `data/memory.json`
- With files: Combines multiple conversation files in order
- Preserves system prompt from current template

**Examples:**

```bash
/load                              # Load default
/load data/memory-20251209.json   # Load specific file
```

### /clear

Clear conversation history and archive current session.

**Syntax:**

```bash
/clear
```

**Behavior:**

- Archives current context to `data/memory-clear-YYYYMMDDTHHMMSS.json`
- Resets to system prompt only
- Auto-saves empty state

### /trim

Manually trim context to fit token limits.

**Syntax:**

```bash
/trim
```

**Behavior:**

- Keeps system prompt + 10 most recent messages
- Target: 75% of model's context window (default 6000 tokens)
- Auto-triggers when context exceeds limit

**Output:**

```
✂️  Context trimmed to N messages
```

### /context

Display current conversation state.

**Syntax:**

```bash
/context          # Full display
/context brief    # Truncated preview
```

**Output:**

```
📋 CURRENT CONTEXT
Total messages: 15 | Estimated tokens: 3420

[0] SYSTEM (~450 tokens):
  You are a helpful assistant...

[1] USER (~120 tokens):
  How do I configure memory storage?
```

## Usage Examples

### Session Management

```bash
💬 You: /save
💾 Memory saved to data/memory.json

💬 You: /clear
📦 Previous conversation archived to data/memory-clear-20251209T153045.json
🗑️  Context cleared and saved!
```

### Context Monitoring

```bash
💬 You: /context brief

📋 CURRENT CONTEXT
Total messages: 8 | Estimated tokens: 2150

📊 Messages: 8 | Tokens: 2150/6000 (35.8%)
```

### Token Management

```bash
💬 You: /trim
✂️  Context trimmed to 11 messages

📊 Messages: 11 | Tokens: 4980/6000 (83.0%)
⚠️  Context nearly full - will auto-trim on next message
```

## Troubleshooting

**"Context file corrupted and no backup available"**

- Start fresh with `/clear`

**Auto-trim triggering too often**

- Reduce conversation length
- Use `/clear` to archive and reset
- Consider shorter system prompts
