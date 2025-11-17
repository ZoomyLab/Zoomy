# .devcontainer/firedrake/Dockerfile

# FROM firedrakeproject/firedrake:latest
FROM stefanofochesatto/viamr:latest

# --- Fix apt permission issues ---
USER root
RUN mkdir -p /var/lib/apt/lists/partial && \
    chmod -R 755 /var/lib/apt/lists

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

RUN bash -lc "source /home/firedrake/firedrake/bin/activate && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      zoomy-core \
      jupyterlab \
      notebook \
      ipywidgets \
      matplotlib \
      pandas && \
    git clone https://github.com/StefanoFochesatto/VI-AMR.git /workspace/VI-AMR && \
    pip install --no-cache-dir --break-system-packages -e /workspace/VI-AMR"

# Ensure Firedrake environment is activated on login
RUN echo "source /home/firedrake/firedrake/bin/activate" >> /etc/bash.bashrc


USER $USERNAME
WORKDIR /workspace
