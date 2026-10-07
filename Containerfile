FROM ghcr.io/astral-sh/uv:0.12.15 AS uv
FROM python:3.12-slim
COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --locked --no-dev --no-editable
ENV PATH="/app/.venv/bin:$PATH"
USER 1001
CMD ["python", "-m", "mobility_ai.capstone.app", "--help"]
