# AltBots Tearsheet Pipeline

> **Institutional-grade manager tearsheet generation for alternative investment research — powered by AI agents, served via MCP.**

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP Server](https://img.shields.io/badge/MCP-Live%20on%20Smithery-green)](https://smithery.ai/servers/ndatta18-b06a/altbots)
[![Status](https://img.shields.io/badge/status-production-brightgreen)]()

---

## What Is This?

AltBots Tearsheet is an end-to-end AI pipeline that generates structured, institutional-quality research tearsheets on alternative investment managers — hedge funds, private equity firms, and family offices — from publicly available data and proprietary intelligence signals.

Built by a former Managing Director with 20+ years of institutional buy-side experience (2,000+ manager reviews, $9B+ deployed), this system encodes the actual diligence framework used at multi-billion-dollar investment offices into a fully automated, agent-driven research workflow.

**Sample output:** [Meridian Capital Management LP — Tearsheet PDF](./samples/meridian-capital-tearsheet.pdf)

---

## Architecture

```
Data Sources (SEC EDGAR, web, news, filings)
        │
        ▼
┌─────────────────────┐
│   Firecrawl Scraper │  ← structured web extraction
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Agent Fleet       │
│  ┌───────────────┐  │
│  │ Carl          │  │  ← equity & fund research
│  │ Scout         │  │  ← due diligence signals
│  │ Tony          │  │  ← chief of staff / orchestration
│  │ Alice         │  │  ← client services / IR
│  └───────────────┘  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Assembler         │  ← normalizes & structures data
│   - Fund Profile    │     sanctions, service providers,
│   - Performance     │     performance, AUM, strategy
│   - Risk Signals    │
│   - Service Provs.  │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  ReportLab PDF      │  ← institutional formatting
│  Formatter          │
└────────┬────────────┘
         │
         ▼
     tearsheet.pdf
```

**Stack:** Python · Anthropic Claude API · OpenAI · Qdrant · n8n · Firecrawl · ReportLab · FastAPI

---

## MCP Server

The tearsheet pipeline is exposed as a live **Model Context Protocol (MCP)** server, making it directly callable from any MCP-compatible AI client (Claude Desktop, Cursor, etc.).

**Connection URL:** `https://altbots--ndatta18-b06a.run.tools`

**Available Tools (4):**
| Tool | Description |
|------|-------------|
| `generate_manager_tearsheet` | Generate an institutional intelligence brief on any SEC-registered alternative investment manager. Returns a 2-page PDF covering SEC filings, personnel, sanctions screening, and social signals. |
| `get_manager_coverage` | Check data coverage and quality score for a manager before generating a full report. Returns per-section coverage assessment and recommendation. |
| `list_available_managers` | List all alternative investment managers indexed in the AltBots database, sorted by AUM descending. |
| `get_pipeline_status` | Get current server health, tool count, managers indexed, and pipeline status. |

**Published on Smithery:** [smithery.ai/servers/ndatta18-b06a/altbots](https://smithery.ai/servers/ndatta18-b06a/altbots)

To connect via Smithery CLI:
```bash
smithery mcp add ndatta18-b06a/altbots
```

Or add directly to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "altbots": {
      "url": "https://altbots--ndatta18-b06a.run.tools"
    }
  }
}
```

---

## Tearsheet Coverage

Each generated tearsheet includes:

- **Fund Identity** — legal name, domicile, registration, key principals
- **Strategy Profile** — asset class, sub-strategy, geographic focus, instruments
- **AUM & Capacity** — reported AUM, share class structure, liquidity terms
- **Performance** — returns summary, drawdown profile, Sharpe/Sortino where available
- **Service Providers** — prime broker, administrator, auditor, legal counsel
- **Sanctions & Flags** — OFAC, regulatory actions, litigation signals
- **Intelligence Signals** — 13 proprietary signal categories including personnel changes, redemption risk, and operational flags

---

## Quickstart

```bash
git clone https://github.com/NeilD-NYC/altbots-tearsheet.git
cd altbots-tearsheet
pip install -r requirements.txt
cp .env.example .env  # add your API keys
python generate_tearsheet.py --manager "Meridian Capital Management LP"
```

Output: `./output/meridian-capital-tearsheet.pdf`

---

## Project Background

This system was designed and built by **Neil Datta**, founder of AltBots and NKD Advisory LLC. Prior to this, Neil served as Managing Director and Global Head of Risk & Diligence at Optima/Forbes Family Trust ($24B multi-family office), where he reviewed 2,000+ alternative investment managers and deployed $9B+ across asset classes.

AltBots encodes that institutional diligence framework into an AI-native research infrastructure that any allocator or investor relations professional can leverage at scale.

**AltBots:** [altbots.io](https://altbots.io)  
**LinkedIn:** [in/neildatta](https://linkedin.com/in/neildatta)

---

## Roadmap

- [x] Core tearsheet PDF generation
- [x] ReportLab formatter
- [x] MCP server with 4 tools
- [x] SSE endpoint live
- [x] Smithery marketplace listing
- [ ] Stripe + API key auth (in progress)
- [ ] Sanctions module (assembler)
- [ ] Service provider enrichment
- [ ] Performance data pipeline
- [ ] Phase 15 infra hardening

---

## License

MIT — see [LICENSE](./LICENSE)






- Free tier: coverage check and manager list
- Pro: full tearsheet generation ($149/month)
- Enterprise: unlimited, white-label, webhooks

## Sample Report

[View sample report on Meridian Capital Management](https://api.altbots.io/sample/AltBots_Sample_Report.pdf)

## Legal

Research signals, not investment advice. 
AltBots and NKD Advisory LLC are not registered investment advisers.
Full disclosures included in every report.

---
Built by NKD Advisory LLC | neil@altbots.io | altbots.io
