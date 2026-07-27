"""Coupled-case contract generator — build the shared ``precice-config.xml``
for a 2-participant Zoomy coupling.

This is the *coupling contract*: the exchanged unit-zeta column profile
``[b, h, u, v, w, p]``, the participant meshes, the nearest-neighbor mapping and
the ``m2n:sockets`` exchange over a shared handshake directory. It is the
physics-specific, load-bearing part that the GUI's placeholder precice-config
stubs out (see model-config-widget.placeholderPreciceConfig).

Ported from the thesis coupling ``generate.py`` XML assembly
(notebooks/coupling/cases/{sme0_sme1,sme_vof}). Verified with preCICE's own
``precice-config-validate`` (SME<->SME).

NOT here (still owned by the solver build): expanding each participant's
``case.py`` spec into a full OpenFOAM case tree (mesh + 0/ fields +
system/controlDict with ``preciceParticipant``/``preciceConfig``), and the
``run.sh`` that launches both participants. Those need the zoomyFoam build to
verify and are tracked separately.
"""
from __future__ import annotations

#: the zeta-column contract — ALWAYS the full profile (one sample per interface
#: face), never the level-0 single-sample shortcut.
PROFILE = ("b", "h", "u", "v", "w", "p")


def _indent(lines, n=2):
    pad = " " * n
    return "\n".join(pad + ln if ln else ln for ln in lines)


def make_sme_sme_config(names=("Sme0", "Sme1"), exchange_directory=".",
                        scheme="parallel-explicit", time_window=5e-4,
                        max_time=30.0, dimensions=3):
    """precice-config.xml for an SME(i) <-> SME(j) coupling (both zoomyFoam).

    Mirrors ``notebooks/coupling/cases/sme0_sme1/precice-config.xml``: each
    participant provides its own mesh, writes its profile ``[x]_1`` / ``[x]_2``
    and reads the peer's, mapped nearest-neighbor.
    """
    p0, p1 = names
    data = [f"{f}_1" for f in PROFILE] + [f"{f}_2" for f in PROFILE]
    w1 = [f"{f}_1" for f in PROFILE]
    w2 = [f"{f}_2" for f in PROFILE]

    def mesh(name):
        return [f'<mesh name="{name}" dimensions="{dimensions}">',
                *[f'  <use-data name="{d}"/>' for d in data], '</mesh>']

    def participant(name, mine, peer_mesh, peer, write, read):
        lines = [f'<participant name="{name}">',
                 f'  <provide-mesh name="{mine}"/>',
                 f'  <receive-mesh name="{peer_mesh}" from="{peer}"/>']
        lines += [f'  <write-data name="{d}" mesh="{mine}"/>' for d in write]
        lines += [f'  <read-data name="{d}" mesh="{mine}"/>' for d in read]
        lines += [f'  <mapping:nearest-neighbor direction="read" from="{peer_mesh}" '
                  f'to="{mine}" constraint="consistent"/>', '</participant>']
        return lines

    exchanges = ([f'<exchange data="{d}" mesh="Mesh0" from="{p0}" to="{p1}"/>' for d in w1]
                 + [f'<exchange data="{d}" mesh="Mesh1" from="{p1}" to="{p0}"/>' for d in w2])

    body = ['<?xml version="1.0" encoding="UTF-8" ?>', '<precice-configuration>', '']
    body += [f'  <data:scalar name="{d}"/>' for d in data] + ['']
    body += ['  ' + ln for ln in mesh("Mesh0")] + ['']
    body += ['  ' + ln for ln in mesh("Mesh1")] + ['']
    body += ['  ' + ln for ln in participant(p0, "Mesh0", "Mesh1", p1, w1, w2)] + ['']
    body += ['  ' + ln for ln in participant(p1, "Mesh1", "Mesh0", p0, w2, w1)] + ['']
    body += [f'  <m2n:sockets acceptor="{p0}" connector="{p1}" '
             f'exchange-directory="{exchange_directory}"/>', '']
    body += [f'  <coupling-scheme:{scheme}>',
             f'    <participants first="{p0}" second="{p1}"/>',
             f'    <max-time value="{max_time}"/>',
             f'    <time-window-size value="{time_window}"/>',
             *['    ' + e for e in exchanges],
             f'  </coupling-scheme:{scheme}>']
    body += ['</precice-configuration>', '']
    return "\n".join(body)


def make_coupled_precice_config(participants, **kw):
    """Dispatch on the participant types. ``participants`` is a list of
    ``{"name": str, "type": "sme"|"vof"}`` (2 entries for now).

    Only the SME<->SME combo is generated + validated so far; SME<->VOF (the
    heterogeneous inlet-profile contract in sme_vof/generate.py) is not yet
    ported — raise rather than emit an unverified contract.
    """
    types = tuple(p.get("type", "sme") for p in participants)
    names = tuple(p["name"] for p in participants)
    if len(participants) == 2 and types == ("sme", "sme"):
        return make_sme_sme_config(names=names, **kw)
    raise NotImplementedError(
        f"coupled precice-config for participant types {types} is not ported yet "
        "(only sme<->sme). See sme_vof/generate.py for the heterogeneous contract."
    )
