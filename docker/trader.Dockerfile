# Multi-stage build: uv resolves and installs into a venv in the builder
# stage, and only the venv plus source are copied into the runtime image —
# no compiler toolchain or uv itself ships in the final image.

FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml ./
COPY src ./src

RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install --no-cache .

FROM python:3.11-slim-bookworm AS runtime

RUN groupadd -g 10001 aitrader && \
    useradd -u 10001 -g aitrader -m -s /usr/sbin/nologin aitrader

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY src ./src
COPY pyproject.toml ./

RUN mkdir -p /app/data && chown -R aitrader:aitrader /app

USER aitrader

ENTRYPOINT ["python", "-m", "aitrader"]
CMD ["run"]
