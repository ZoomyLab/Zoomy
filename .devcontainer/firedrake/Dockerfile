# .devcontainer/firedrake/Dockerfile
FROM firedrakeproject/firedrake:latest

# --- Install system dependencies ---
RUN apt-get update && \
    apt-get install -y git openssh-client sudo && \
    rm -rf /var/lib/apt/lists/*

# --- Create (or reuse) vscode user safely ---
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN set -eux; \
    # Check if a user with UID 1000 already exists (e.g. 'firedrake') and reuse it
    EXISTING_USER=$(getent passwd $USER_UID | cut -d: -f1 || true); \
    if [ -n "$EXISTING_USER" ]; then \
        echo "Reusing existing user: $EXISTING_USER"; \
        usermod -l $USERNAME $EXISTING_USER || true; \
        groupmod -n $USERNAME $(id -gn $USERNAME || id -gn $EXISTING_USER) || true; \
    else \
        groupadd -g $USER_GID $USERNAME || true; \
        useradd -m -s /bin/bash -u $USER_UID -g $USER_GID $USERNAME; \
    fi; \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME; \
    chmod 0440 /etc/sudoers.d/$USERNAME; \
    mkdir -p /workspace && chown -R $USERNAME:$USERNAME /workspace

# --- Install Python dependencies ---
RUN python3 -m pip install --no-cache-dir \
      zoomy-core \
      jupyterlab \
      notebook \
      ipywidgets \
      matplotlib \
      pandas

USER $USERNAME
WORKDIR /workspace
