FROM ghcr.io/zoomylab/zoomy_firedrake:latest

# Install minimal X11 libs so Firedrake's PETSc/Xlib detection works
RUN apt-get update && apt-get install -y \
    libx11-6 \
    libxext6 \
    libxt6 \
    libxrender1 \
    libxft2 \
    libxpm4 \
    && rm -rf /var/lib/apt/lists/*

# Copy all library modules
COPY ./library /tmp/library

# Install them in editable mode (deps included)
RUN for d in /tmp/library/*; do \
      if [ -f "$d/pyproject.toml" ]; then \
        pip install -e "$d"; \
      fi; \
    done
