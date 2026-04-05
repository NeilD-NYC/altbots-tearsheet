# Project

## Structure

```
src/
  core/         # Data fetching, scoring, and assembly
  renderers/    # Output renderers (PDF, etc.)
  mcp/          # MCP server and auth
  skill/        # Skill entrypoint
  ingestion/    # Email and document ingestion
  api/          # API endpoints
config/         # Application settings
tests/          # Test suite
```

## Setup

```bash
pip install -r requirements.txt
cp .env .env.local  # add your secrets
```
