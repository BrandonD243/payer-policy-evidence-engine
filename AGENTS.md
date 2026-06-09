# AGENTS.md

## Project conventions
- This is a FastAPI prior authorization evidence engine.
- Keep business logic in services, not route handlers.
- Avoid hardcoding patient data except in test fixtures.
- Do not introduce real PHI.
- Use fictional test data only.
- Preserve existing API behavior unless the task explicitly asks for changes.
- Add or update tests for new service logic.
- Run the relevant tests before finalizing changes.

## Prior authorization architecture
The system should treat PDF generation, email submission, portal field preparation, and API payload generation as separate output/submission adapters. The core PA case object should be independent from any one output method.
