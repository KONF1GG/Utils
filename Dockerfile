FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

COPY . /app/

RUN uv venv


RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project


RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

EXPOSE 8000


CMD ["uv", "run", "main.py"]