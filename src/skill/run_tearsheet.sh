#!/bin/bash
set -e
cd ~/tearsheet-project
source venv/bin/activate
MANAGER="$1"
FORMAT="${2:-pdf}"
if [ -z "$MANAGER" ]; then
  echo "Usage: run_tearsheet.sh 'Manager Name' [pdf|md]"
  exit 1
fi
echo "Running coverage check for: $MANAGER"
python main.py "$MANAGER" --format $FORMAT
OUTPUT=$(ls -t output/*.pdf | head -1)
cp "$OUTPUT" /var/www/html/sample/$(basename "$OUTPUT")
FILENAME=$(basename "$OUTPUT")
echo "Done. Report available at:"
echo "https://api.altbots.io/sample/$FILENAME"
