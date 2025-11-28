#!/bin/bash

# Setup script for Retail Analytics Copilot

echo "Setting up Retail Analytics Copilot"

# Create directories
mkdir -p data docs agent/rag agent/tools

# Download Northwind database
echo "Downloading Northwind database"
curl -L -o data/northwind.sqlite \
  https://raw.githubusercontent.com/jpwhite3/northwind-SQLite3/main/dist/northwind.db

# Create document corpus
echo "Creating documents corpus"

cat > docs/marketing_calendar.md << 'EOF'
# Northwind Marketing Calendar (1997)

## Summer Beverages 1997
- Dates: 1997-06-01 to 1997-06-30
- Notes: Focus on Beverages and Condiments.

## Winter Classics 1997
- Dates: 1997-12-01 to 1997-12-31
- Notes: Push Dairy Products and Confections for holiday gifting.
EOF

cat > docs/kpi_definitions.md << 'EOF'
# KPI Definitions

## Average Order Value (AOV)
- AOV = SUM(UnitPrice * Quantity * (1 - Discount)) / COUNT(DISTINCT OrderID)

## Gross Margin
- GM = SUM((UnitPrice - CostOfGoods) * Quantity * (1 - Discount))
- If cost is missing, approximate with category-level average (document your approach).
EOF

cat > docs/catalog.md << 'EOF'
# Catalog Snapshot

- Categories include Beverages, Condiments, Confections, Dairy Products, Grains/Cereals, Meat/Poultry, Produce, Seafood.
- Products map to categories as in the Northwind DB.
EOF

cat > docs/product_policy.md << 'EOF'
# Returns & Policy

- Perishables (Produce, Seafood, Dairy): 3–7 days.
- Beverages unopened: 14 days; opened: no returns.
- Non-perishables: 30 days.
EOF

# Create __init__.py files
touch agent/__init__.py
touch agent/rag/__init__.py
touch agent/tools/__init__.py

echo "downloading phi 3.5"
ollama pull phi3.5:3.8b-mini-instruct-q4_K_M 

echo "Setup complete!"
