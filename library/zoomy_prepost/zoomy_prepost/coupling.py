"""Coupled-case contract generator — build the shared ``precice-config.xml``
for a 2-participant Zoomy coupling.

This is the *coupling contract*: the exchanged unit-zeta column profile
``[b, h, u, v, w, p]``, the participant meshes, the nearest-neighbor mapping and
the ``m2n:sockets`` exchange over a shared handshake directory. It is the
physics-specific, load-bearing part that the GUI's placeholder precice-config
stubs out (see model-config-widget.placeholderPreciceConfig).

Ported from the thesis coupling ``generate.py`` XML assembly
(notebooks/coupling/cases/{sme0_sme1,sme_vof}). Both SME<->SME and SME<->VOF
share ONE topology: each participant provides its mesh, writes its own suffixed
profile ``[x]_<suffix>`` and reads the peer's, mapped nearest-neighbor. Verified
with preCICE's own ``precice-config-validate``.

Also here: ``build_coupled_bundle`` — the template-based OF-case expansion the
GUI "Run all" path assembles (participant cases as siblings under the shared
coupling folder, sharing one generated precice-config.xml).
"""
from __future__ import annotations

import os
import re
import shutil

#: the zeta-column contract — ALWAYS the full profile (one sample per interface
#: face), never the level-0 single-sample shortcut.
PROFILE = ("b", "h", "u", "v", "w", "p")


def _two_participant_config(p0, p1, exchange_directory=".",
                           scheme="parallel-explicit", time_window=5e-4,
                           max_time=30.0, dimensions=3):
    """Symmetric profile-exchange config. p0/p1 are dicts
    ``{"name","mesh","suffix"}`` (Sme0/Mesh0/1 <-> Sme1/Mesh1/2, or
    Swe/SweMesh/S <-> Vof/VofInletMesh/V)."""
    d0 = [f"{f}_{p0['suffix']}" for f in PROFILE]
    d1 = [f"{f}_{p1['suffix']}" for f in PROFILE]
    data = d0 + d1

    def mesh(name):
        return [f'<mesh name="{name}" dimensions="{dimensions}">',
                *[f'  <use-data name="{d}"/>' for d in data], '</mesh>']

    def participant(me, peer, write, read):
        lines = [f'<participant name="{me["name"]}">',
                 f'  <provide-mesh name="{me["mesh"]}"/>',
                 f'  <receive-mesh name="{peer["mesh"]}" from="{peer["name"]}"/>']
        lines += [f'  <write-data name="{d}" mesh="{me["mesh"]}"/>' for d in write]
        lines += [f'  <read-data name="{d}" mesh="{me["mesh"]}"/>' for d in read]
        lines += [f'  <mapping:nearest-neighbor direction="read" '
                  f'from="{peer["mesh"]}" to="{me["mesh"]}" constraint="consistent"/>',
                  '</participant>']
        return lines

    exchanges = ([f'<exchange data="{d}" mesh="{p0["mesh"]}" from="{p0["name"]}" to="{p1["name"]}"/>' for d in d0]
                 + [f'<exchange data="{d}" mesh="{p1["mesh"]}" from="{p1["name"]}" to="{p0["name"]}"/>' for d in d1])

    body = ['<?xml version="1.0" encoding="UTF-8" ?>', '<precice-configuration>', '']
    body += [f'  <data:scalar name="{d}"/>' for d in data] + ['']
    body += ['  ' + ln for ln in mesh(p0['mesh'])] + ['']
    body += ['  ' + ln for ln in mesh(p1['mesh'])] + ['']
    body += ['  ' + ln for ln in participant(p0, p1, d0, d1)] + ['']
    body += ['  ' + ln for ln in participant(p1, p0, d1, d0)] + ['']
    body += [f'  <m2n:sockets acceptor="{p0["name"]}" connector="{p1["name"]}" '
             f'exchange-directory="{exchange_directory}"/>', '']
    body += [f'  <coupling-scheme:{scheme}>',
             f'    <participants first="{p0["name"]}" second="{p1["name"]}"/>',
             f'    <max-time value="{max_time}"/>',
             f'    <time-window-size value="{time_window}"/>',
             *['    ' + e for e in exchanges],
             f'  </coupling-scheme:{scheme}>']
    body += ['</precice-configuration>', '']
    return "\n".join(body)


def make_sme_sme_config(names=("Sme0", "Sme1"), **kw):
    """SME(i) <-> SME(j) — mirrors sme0_sme1/precice-config.xml."""
    return _two_participant_config({"name": names[0], "mesh": "Mesh0", "suffix": "1"},
                                   {"name": names[1], "mesh": "Mesh1", "suffix": "2"}, **kw)


def make_sme_vof_config(names=("Swe", "Vof"), **kw):
    """SME <-> VOF — mirrors sme_vof/precice-config.xml (SweMesh<->VofInletMesh)."""
    return _two_participant_config({"name": names[0], "mesh": "SweMesh", "suffix": "S"},
                                   {"name": names[1], "mesh": "VofInletMesh", "suffix": "V"}, **kw)


def make_coupled_precice_config(participants, **kw):
    """Dispatch on participant types. ``participants`` is a list of
    ``{"name": str, "type": "sme"|"vof"}`` (2 entries)."""
    types = tuple(p.get("type", "sme") for p in participants)
    names = tuple(p["name"] for p in participants)
    if len(participants) != 2:
        raise NotImplementedError("only 2-participant couplings are generated so far")
    if types == ("sme", "sme"):
        return make_sme_sme_config(names=names, **kw)
    if types == ("sme", "vof"):
        return make_sme_vof_config(names=names, **kw)
    if types == ("vof", "sme"):
        return make_sme_vof_config(names=(names[1], names[0]), **kw)
    raise NotImplementedError(f"coupled precice-config for types {types} not ported")


# --------------------------------------------------------------------------- #
# OF-case expansion (template-based) — the coupled bundle the GUI "Run all"
# assembles: participant cases as siblings under the shared coupling folder
# (= the exchange-directory), sharing one generated precice-config.xml.
# --------------------------------------------------------------------------- #
def build_participant_case(dest, template_case, participant_name, config_path,
                           end_time=None):
    """Expand one participant OF case from a template (copy 0/ constant/ system/),
    rewiring its controlDict to this coupling's participant name + shared config.

    ``template_case`` is a ready OF participant case (the thesis
    sme0_sme1/part1 or sme_vof/run/swe_case) — the emitters in the thesis
    generate.py already produce exactly this (mesh + 0/ fields + controlDict with
    precice* keys). Copying keeps the geometry/IC/ZSamples; only the participant
    identity + config path (+ optional end_time) are rewritten."""
    os.makedirs(dest, exist_ok=True)
    for sub in ("0", "constant", "system"):
        src = os.path.join(template_case, sub)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(dest, sub), dirs_exist_ok=True)
    cd = os.path.join(dest, "system", "controlDict")
    txt = open(cd).read()
    txt = re.sub(r'preciceParticipant\s+\w+;', f'preciceParticipant {participant_name};', txt)
    txt = re.sub(r'preciceConfig\s+"[^"]*";', f'preciceConfig "{config_path}";', txt)
    if end_time is not None:
        txt = re.sub(r'endTime\s+[0-9.]+;', f'endTime {end_time};', txt)
    open(cd, "w").write(txt)
    return dest


def build_coupled_bundle(coupling_dir, participants, exchange_directory=None,
                         end_time=None, **cfg_kw):
    """Assemble a runnable coupled bundle under ``coupling_dir``: write the shared
    precice-config.xml and expand each participant OF case as a sibling folder.

    ``participants``: list of ``{"name","type","template","case_name"}``. Returns
    ``{"config": path, "cases": [(name, dir), ...]}``. The launcher runs each
    ``<case_dir>`` (its controlDict names the participant + config) in the sif,
    all sharing ``coupling_dir`` as the exchange-directory."""
    os.makedirs(coupling_dir, exist_ok=True)
    exch = exchange_directory or coupling_dir
    cfg = make_coupled_precice_config(
        participants, exchange_directory=exch,
        max_time=(end_time if end_time is not None else 30.0), **cfg_kw)
    config_path = os.path.join(coupling_dir, "precice-config.xml")
    open(config_path, "w").write(cfg)
    cases = []
    for p in participants:
        case_dir = os.path.join(coupling_dir, p.get("case_name", p["name"] + "_case"))
        build_participant_case(case_dir, p["template"], p["name"], config_path,
                               end_time=end_time)
        cases.append((p["name"], case_dir))
    return {"config": config_path, "cases": cases}
