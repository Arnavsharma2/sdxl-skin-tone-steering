# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 python:3.12.11-slim-bookworm@sha256:c00fc7b44d844b6da22861ec24af43968a5200eac4ec607b4725d585165d6b49

ARG DEBIAN_SNAPSHOT=20260801T000000Z
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONHASHSEED=0 \
    PYTHONPATH=/workspace \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/artifacts/torch

RUN rm -f /etc/apt/sources.list.d/debian.sources \
    && printf '%s\n' \
      "deb [check-valid-until=no] https://snapshot.debian.org/archive/debian/${DEBIAN_SNAPSHOT} bookworm main" \
      "deb [check-valid-until=no] https://snapshot.debian.org/archive/debian-security/${DEBIAN_SNAPSHOT} bookworm-security main" \
      > /etc/apt/sources.list \
    && apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements/evaluation-linux-py312.lock /tmp/evaluation-linux-py312.lock
RUN python -m pip install \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      --require-hashes \
      -r /tmp/evaluation-linux-py312.lock

COPY . /workspace

CMD ["python", "-m", "scripts.validate_readiness", "--strict"]
