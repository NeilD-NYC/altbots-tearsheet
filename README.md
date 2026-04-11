# AltBots — Fund Manager Intelligence MCP

Institutional-grade research reports on alternative investment managers. 
SEC filings, personnel intelligence, sanctions screening, social signals. 
Generated in under 60 seconds.

Built by a former Global Head of Risk and Diligence with 20 years and 
2,000+ manager evaluations.

## MCP Server

**Endpoint:** `https://api.altbots.io/sse`

**Tools:**

| Tool | Description |
|---|---|
| `generate_manager_tearsheet` | Generate a full intelligence brief on any SEC-registered manager. Returns PDF URL, JSON or markdown. |
| `get_manager_coverage` | Check data coverage before generating. Returns score and recommendation. |
| `list_available_managers` | List indexed managers sorted by AUM. |
| `get_pipeline_status` | Server health and pipeline status. |

## Quick Start

Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "altbots": {
      "url": "https://api.altbots.io/sse",
      "transport": "sse"
    }
  }
}
```

## Example Usage

```
Generate a tearsheet on Bridgewater Associates
```

```
Check coverage for Viking Global Investors
```
## Output

Each report includes:
- SEC Form ADV summary (AUM, strategy, clients, fees)
- 13F top holdings and concentration
- Key personnel with FINRA BrokerCheck status
- Sanctions screening (OFAC, EU, UN, HMT, FinCEN)
- Social and media signals (last 90 days)
- Red flags and regulatory observations
- Full legal disclosure block

## Pricing

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
