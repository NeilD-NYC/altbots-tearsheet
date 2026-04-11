---
name: altbots-tearsheet
description: >
  Generates an institutional-grade fund manager
  intelligence brief. Use when asked for:
  tearsheet, one-pager, manager summary,
  ODD report, intelligence brief, research
  report, generate report on [any manager name].

  Produces a 2-page PDF covering SEC filings,
  personnel, social signals and legal disclosures.
  Output: ~/tearsheet-project/output/[manager].pdf
  NOT investment advice. Research signals only.
---

# AltBots Tearsheet Skill

This skill generates institutional-grade fund manager intelligence briefs via the AltBots pipeline.

## When This Skill Applies

Activate when the user requests any of:
- A tearsheet, one-pager, or manager summary
- An ODD report or operational due diligence brief
- An intelligence brief or research report
- "Generate a report on [manager name]"
- Any request that names a fund manager alongside words like report, brief, summary, profile

## How to Execute

Run the tearsheet script with the manager name:

```bash
bash ~/tearsheet-project/src/skill/run_tearsheet.sh "Manager Name"
```

Or invoke the pipeline directly for more control:

```bash
cd ~/tearsheet-project && source venv/bin/activate
python main.py "Manager Name" --format pdf
```

For enriched output (overlays data from data/enriched/):
```bash
python main.py "Manager Name" --format pdf --enrich
```

## Output

- PDF saved to: `~/tearsheet-project/output/<SLUG>_<DATE>.pdf`
- PDF also copied to: `/var/www/html/sample/<filename>.pdf`
- Accessible at: `https://api.altbots.io/sample/<filename>.pdf`

## Coverage Check

Before rendering, the pipeline prints a coverage table showing which data sources are available:
- SEC ADV filing
- Form 13F holdings
- Team roster profiles
- Social signals (Exa.ai)

## Notes

- NOT investment advice. Research signals only.
- For unknown managers, use `--enrich --manual` to interactively enter data
- PDF is a 2-page institutional brief with legal disclosures on page 2
