"""engine/ — Recommendation and decisioning layer for partswatch-ai.

Modules
-------
transfer.py
    Pure-computation module.  Identifies locations with excess stock of a given
    SKU that can fulfill an inter-location transfer.  No database I/O.

reorder.py
    Orchestrator.  Converts ML forecasts into purchase-order and transfer
    recommendations and writes them to reorder_recommendations.
"""
