FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    darcs gcc make gawk flex bison && rm -rf /var/lib/apt/lists/*

ENV BASILISK=/basilisk/src
ENV PATH=$PATH:$BASILISK

RUN darcs get http://basilisk.fr/basilisk /basilisk && \
    cd /basilisk/src && \
    cp config.gcc config && \
    make -C ast && \
    make qcc

# Inject our custom VTK/Series exporter directly into the Basilisk library
COPY to_vtk.h /basilisk/src/to_vtk.h

COPY run_case.sh /usr/local/bin/run_case
RUN chmod +x /usr/local/bin/run_case

WORKDIR /workspace
ENTRYPOINT ["run_case"]
