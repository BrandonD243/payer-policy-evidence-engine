# Payer Policy Reasoning Engine

A knowledge-graph–driven system for evaluating medical prior authorization
decisions (starting with MRI Lumbar Spine).

## Architecture Overview

This project separates concerns into three layers:

1. **Knowledge Modeling**
   - Universal clinical concepts and archetypes
   - Policy-specific clauses and criteria
   - Compiled into static JSON knowledge graphs

2. **Runtime Compilation**
   - Patient documents are extracted into structured facts
   - Only relevant policy and KG slices are compiled per case

3. **Agentic Reasoning**
   - A decision agent evaluates approval/denial
   - Outputs are validated against expected outcomes

## Key Directories

