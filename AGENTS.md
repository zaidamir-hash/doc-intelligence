# AGENTS.md

## Project Context

This is a learning-focused software/ML project.

The authoritative implementation roadmap is:

`docs/PROJECT_PLAN.md`

Read the relevant section of the project plan before beginning
any major implementation task.

## Development Workflow

- Work on only the phase or task explicitly requested.
- Do not automatically begin subsequent phases.
- Before significant changes, inspect the existing implementation.
- Follow existing architecture and conventions where reasonable.
- Do not make unrelated changes.
- Prefer small, reviewable changes over large rewrites.
- Reuse existing functionality rather than duplicating it.
- Do not introduce new dependencies unless they are justified.

## Learning

The developer is using this project to learn.

- Explain important architectural and technical decisions.
- Explain non-obvious code and ML concepts.
- Do not hide important implementation details behind unnecessary abstractions.
- When appropriate, allow the developer to implement educationally
  valuable pieces instead of automatically writing everything.
- When reviewing developer-written code, explain problems before
  replacing the implementation.

## Code Quality

- Prefer simple, readable, maintainable code.
- Use descriptive names.
- Keep functions/classes focused.
- Add type hints where appropriate.
- Avoid premature abstractions and over-engineering.
- Handle errors explicitly where appropriate.
- Follow the project's existing formatting and style conventions.

## Testing

After implementing a meaningful change:

- Run relevant existing tests.
- Add focused tests for new behavior where appropriate.
- Do not claim tests passed unless they were actually executed.
- Report failing tests rather than hiding or bypassing them.
- Do not modify tests merely to make an incorrect implementation pass.

## ML / Evaluation

- Keep training, validation, and test data properly separated.
- Avoid data leakage.
- Keep experiments reproducible where practical.
- Record important experiment parameters.
- Do not fabricate metrics or experimental results.
- Do not change evaluation methodology solely to improve reported results.

## Project Plan

After completing a phase:

1. Compare the implementation against `docs/PROJECT_PLAN.md`.
2. Verify its acceptance criteria.
3. Run relevant tests.
4. Mark checklist items complete only when actually completed.
5. Record meaningful deviations from the original plan.
6. Do not begin the next phase unless explicitly requested.

## Git

- Do not commit or push unless explicitly requested.
- Never force-push unless explicitly requested.
- Do not delete branches unless explicitly requested.
- Before a requested commit, review the current diff and test status.
- Use clear, descriptive commit messages.

## Security

- Never commit API keys, passwords, tokens, or other secrets.
- Keep secrets in environment variables.
- Keep `.env` excluded from Git.
- Update `.env.example` when new configuration variables are introduced.
