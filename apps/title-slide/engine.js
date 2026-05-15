/**
 * apps/title-slide/engine.js — rectangular WebGL2 SWE engine.
 *
 * Algorithm: MUSCL reconstruction + HLLC + 2-stage Heun, identical to
 * the swe-game GPU engine; only the surrounding domain/render layer is
 * specialised for title-slide use:
 *
 *   • Rectangular grid:           ``(NX+2) × (NY+2)`` texture, configurable
 *                                 NX,NY (default 16:9 — 480 × 270).
 *   • Boundary layout:            inflow on a fraction of the left edge,
 *                                 outflow on a fraction of the right edge,
 *                                 reflective wall everywhere else.
 *                                 No internal walls, no user-drawn mask.
 *   • Render:                     interior cells only (no ghost border)
 *                                 → outer walls are never visible.
 *   • Fixed Δt:                   ``cfl · dx / sMax``; default sMax = 3 m/s.
 *   • Q(t):                       same 20-point timeline parameter block as
 *                                 the game, but timeline times are remapped
 *                                 to span ``[0, endTime]``.
 *
 * Loads the generated kernels from ``window.SWE2D`` (set by a classic
 * ``<script src="./generated/swe2d_glsl.js">`` tag in index.html).
 */

const VS_QUAD = `#version 300 es
in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }
`;

function buildSimShader(kernels, nParams) {
  return `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uA;
uniform sampler2D uB;
uniform sampler2D uBmask;
uniform float uDt, uDx, uTime;
uniform int uStage;
uniform float uP[${nParams}];
out vec4 oQ;

// ── Generated SWE kernels ─────────────────────────────────────────
${kernels}

// Boundary-tag → kernel index (BoundaryConditions sorts tags
// inflow < outflow < wall in the meta JSON).
const int BC_INFLOW  = 0;
const int BC_OUTFLOW = 1;
const int BC_WALL    = 2;

vec3 readA(int i, int j) { return texelFetch(uA, ivec2(i, j), 0).xyz; }
vec3 readB(int i, int j) { return texelFetch(uB, ivec2(i, j), 0).xyz; }
int readBT(int i, int j) { return int(texelFetch(uBmask, ivec2(i, j), 0).r * 255.0 + 0.5); }

float mm1(float a, float b) { if (a*b <= 0.0) return 0.0; return abs(a) < abs(b) ? a : b; }
vec3 mm3(vec3 a, vec3 b) { return vec3(mm1(a.x,b.x), mm1(a.y,b.y), mm1(a.z,b.z)); }

vec3 slopeX(int i, int j) {
  return mm3(readA(i,j) - readA(i-1,j), readA(i+1,j) - readA(i,j));
}
vec3 slopeY(int i, int j) {
  return mm3(readA(i,j) - readA(i,j-1), readA(i,j+1) - readA(i,j));
}

void toArr(vec3 v, out float a[3]) { a[0]=v.x; a[1]=v.y; a[2]=v.z; }
vec3 toVec(float a[3]) { return vec3(a[0], a[1], a[2]); }

vec3 numFluxX(vec3 L, vec3 R) {
  const float n[2] = float[2](1.0, 0.0);
  float QL[3]; float QR[3]; float res[3];
  toArr(L, QL); toArr(R, QR);
  numerical_flux(QL, QR, uP, n, res);
  return toVec(res);
}
vec3 numFluxY(vec3 L, vec3 R) {
  const float n[2] = float[2](0.0, 1.0);
  float QL[3]; float QR[3]; float res[3];
  toArr(L, QL); toArr(R, QR);
  numerical_flux(QL, QR, uP, n, res);
  return toVec(res);
}

vec3 edgeGhost(int bcIdx, vec3 qIn, float n[2]) {
  float Qin[3]; float res[3];
  float X[3] = float[3](0.0, 0.0, 0.0);
  toArr(qIn, Qin);
  if (bcIdx == BC_INFLOW)       bc_inflow(uTime, X, uDx, Qin, uP, n, res);
  else if (bcIdx == BC_OUTFLOW) bc_outflow(uTime, X, uDx, Qin, uP, n, res);
  else                          bc_wall(uTime, X, uDx, Qin, uP, n, res);
  return toVec(res);
}

void main() {
  ivec2 cell = ivec2(gl_FragCoord.xy);
  int i = cell.x, j = cell.y;
  int bt = readBT(i, j);
  vec3 qC = readA(i, j);

  // Ghost cell: dispatch to the appropriate BC kernel using the
  // interior neighbour as the "minus" state.
  if (bt != 0) {
    int iIn = i, jIn = j;
    float nx0 = 0.0, ny0 = 0.0;
    if (bt == 1 || bt == 5)      { iIn = i+1; nx0 = -1.0; }
    else if (bt == 2 || bt == 6) { iIn = i-1; nx0 =  1.0; }
    else if (bt == 3)            { jIn = j+1; ny0 = -1.0; }
    else                         { jIn = j-1; ny0 =  1.0; }
    int bcIdx = BC_WALL;
    if (bt == 5) bcIdx = BC_INFLOW;
    else if (bt == 6) bcIdx = BC_OUTFLOW;
    float n[2] = float[2](nx0, ny0);
    oQ = vec4(edgeGhost(bcIdx, readA(iIn, jIn), n), 0.0);
    return;
  }

  // Interior fluid cell: MUSCL reconstruction + numerical flux.
  vec3 sxC = slopeX(i,j), sxE = slopeX(i+1,j), sxW = slopeX(i-1,j);
  vec3 syC = slopeY(i,j), syN = slopeY(i,j+1), syS = slopeY(i,j-1);
  vec3 qE = readA(i+1,j), qW = readA(i-1,j), qN = readA(i,j+1), qS = readA(i,j-1);
  vec3 qCR = qC + 0.5*sxC, qEL = qE - 0.5*sxE;
  vec3 qCL = qC - 0.5*sxC, qWR = qW + 0.5*sxW;
  vec3 qCN = qC + 0.5*syC, qNS = qN - 0.5*syN;
  vec3 qCS = qC - 0.5*syC, qSN = qS + 0.5*syS;
  qCR.x = max(qCR.x, 0.0); qEL.x = max(qEL.x, 0.0);
  qCL.x = max(qCL.x, 0.0); qWR.x = max(qWR.x, 0.0);
  qCN.x = max(qCN.x, 0.0); qNS.x = max(qNS.x, 0.0);
  qCS.x = max(qCS.x, 0.0); qSN.x = max(qSN.x, 0.0);
  vec3 fE = numFluxX(qCR, qEL), fW = numFluxX(qWR, qCL);
  vec3 fN = numFluxY(qCN, qNS), fS = numFluxY(qSN, qCS);
  vec3 rhs = (fW - fE + fS - fN) / uDx;
  vec3 res;
  if (uStage == 0) res = qC + uDt * rhs;
  else { vec3 qN0 = readB(i, j); res = 0.5 * (qN0 + qC + uDt * rhs); }
  if (res.x <= 1e-6) res = vec3(0.0);
  oQ = vec4(res, 0.0);
}
`;
}

const FS_RENDER = `#version 300 es
precision highp float;
precision highp sampler2D;
uniform sampler2D uA;
uniform sampler2D uLUT;
uniform int uMode;          // 0 = h, 1 = |v|
uniform int uNY;
uniform float uScale, uWet;
out vec4 oColor;

vec3 readA(int i, int j) { return texelFetch(uA, ivec2(i, j), 0).xyz; }

float quantize(vec3 q) {
  // uScale = 254 / saturation_value (JS multiplies in).
  // Always maps [0, sat] → [0, 254].
  float val = 0.0;
  if (uMode == 1) {
    if (q.x > uWet) {
      float u_ = q.y/q.x, v_ = q.z/q.x;
      val = sqrt(u_*u_ + v_*v_) * uScale;
    }
  } else {
    val = q.x * uScale;
  }
  return clamp(val, 0.0, 254.0);
}

void main() {
  ivec2 pix = ivec2(gl_FragCoord.xy);
  // Pixel (0,0) is canvas BOTTOM-LEFT in WebGL. We render the interior
  // strip i ∈ [1, NX], j ∈ [1, NY] mapped 1:1 onto the canvas.
  // The toBlob/PNG export then handles the final top-down orientation.
  int ci = pix.x + 1;
  int cj = pix.y + 1;
  float v = quantize(readA(ci, cj));
  oColor = texture(uLUT, vec2((v + 1.0) / 256.0, 0.5));
}
`;

export class TitleSlideEngine {
  constructor(canvas) {
    const gl = canvas.getContext("webgl2", {
      antialias: false, alpha: false,
      premultipliedAlpha: false, preserveDrawingBuffer: true,
    });
    if (!gl) throw new Error("WebGL2 not available");
    if (!gl.getExtension("EXT_color_buffer_float")) {
      throw new Error("EXT_color_buffer_float not available");
    }
    if (!window.SWE2D) throw new Error("generated SWE kernels (window.SWE2D) not loaded");
    this.gl = gl; this.canvas = canvas;
    this.meta = window.SWE2D.meta;
    this.simProg = this._buildProgram(
      VS_QUAD,
      buildSimShader(window.SWE2D.glsl, this.meta.parameterNames.length),
    );
    this.renderProg = this._buildProgram(VS_QUAD, FS_RENDER);
    this._setupQuad();
    this.lutTex = null;
    this.mode = "h";
    this.scale = 0.25;            // colour-scale saturation (m for h, m/s for |v|)
    this.wet = 1e-6;
    this.qCurveData = [[0, 1.0], [1, 1.0]];
    this.time = 0;
    this._configured = false;
  }

  _buildProgram(vsSrc, fsSrc) {
    const gl = this.gl;
    const vs = gl.createShader(gl.VERTEX_SHADER);
    gl.shaderSource(vs, vsSrc); gl.compileShader(vs);
    if (!gl.getShaderParameter(vs, gl.COMPILE_STATUS))
      throw new Error("VS: " + gl.getShaderInfoLog(vs));
    const fs = gl.createShader(gl.FRAGMENT_SHADER);
    gl.shaderSource(fs, fsSrc); gl.compileShader(fs);
    if (!gl.getShaderParameter(fs, gl.COMPILE_STATUS))
      throw new Error("FS: " + gl.getShaderInfoLog(fs));
    const p = gl.createProgram();
    gl.attachShader(p, vs); gl.attachShader(p, fs);
    gl.bindAttribLocation(p, 0, "aPos");
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS))
      throw new Error("link: " + gl.getProgramInfoLog(p));
    return p;
  }

  _setupQuad() {
    const gl = this.gl;
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
      -1,-1, 1,-1, -1,1,  1,-1, 1,1, -1,1,
    ]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  _uloc(prog, name) {
    if (!prog._locs) prog._locs = {};
    if (!(name in prog._locs)) prog._locs[name] = this.gl.getUniformLocation(prog, name);
    return prog._locs[name];
  }

  /**
   * @param {object} cfg
   * @param {number} cfg.nx           interior cells in x
   * @param {number} cfg.ny           interior cells in y
   * @param {number} cfg.endTime      simulation end time (s)
   * @param {number} [cfg.domainWidth] physical domain width in m (default 20)
   * @param {[number,number]} [cfg.inletFracY] opening interval on left edge in [0,1] (default [0.35, 0.65])
   * @param {[number,number]} [cfg.outletFracY] opening interval on right edge in [0,1] (default [0.35, 0.65])
   * @param {number} [cfg.hIn]        target inflow depth (default 0.1 m)
   * @param {number} [cfg.qIn]        base inflow discharge (default 0.01 m²/s)
   * @param {number} [cfg.h0]         initial uniform depth (default 0.01 m)
   * @param {number} [cfg.cfl]        Courant number (default 0.4)
   * @param {number} [cfg.sMax]       assumed max wave speed for fixed dt (default 3 m/s)
   * @param {number[][]} [cfg.qCurve] piecewise-linear Q(t) multiplier as [[t,v],…]
   */
  configure(cfg) {
    const nx = cfg.nx, ny = cfg.ny;
    const domainWidth = cfg.domainWidth ?? 20.0;
    const inletFracY = cfg.inletFracY ?? [0.35, 0.65];
    const outletFracY = cfg.outletFracY ?? [0.35, 0.65];
    const hIn = cfg.hIn ?? 0.1;
    const qIn = cfg.qIn ?? 0.01;
    const h0 = cfg.h0 ?? 0.01;
    const cfl = cfg.cfl ?? 0.4;
    const sMax = cfg.sMax ?? 3.0;

    this.nx = nx; this.ny = ny;
    this.NX_full = nx + 2; this.NY_full = ny + 2;
    this.dx = domainWidth / this.NX_full;
    this.endTime = cfg.endTime;
    this.inletFracY = inletFracY;
    this.outletFracY = outletFracY;
    this.dt = cfl * this.dx / sMax;
    this.canvas.width = nx;
    this.canvas.height = ny;

    // Build the parameter vector and inject geometry-specific overrides
    // (the codegen defaults are the game's, not ours).
    this.params = Float32Array.from(this.meta.parameterDefaults);
    const idxG = this.meta.parameterNames.indexOf("g");
    const idxHIn = this.meta.parameterNames.indexOf("h_in");
    const idxQIn = this.meta.parameterNames.indexOf("q_in");
    if (idxG >= 0) this.params[idxG] = 9.81;
    this.params[idxHIn] = hIn;
    this.params[idxQIn] = qIn;
    // Remap timeline times to [0, endTime].
    const tb = this.meta.timelineBlock;
    for (let i = 0; i < tb.nPoints; i++) {
      this.params[tb.timeStart + i] = this.endTime * i / (tb.nPoints - 1);
    }
    if (cfg.qCurve) this.qCurveData = cfg.qCurve;
    this._applyQCurve();

    this._destroyTextures();
    this._createTextures();
    this._uploadBoundaryMask();
    this._fillInitialState(h0);
    this.time = 0;
    this._configured = true;
  }

  _createTextures() {
    const gl = this.gl;
    const NX = this.NX_full, NY = this.NY_full;
    this.tex = {}; this.fbo = {};
    for (const k of ["A", "B", "C"]) {
      const t = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, NX, NY, 0, gl.RGBA, gl.FLOAT, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      this.tex[k] = t;
      const fb = gl.createFramebuffer();
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE)
        throw new Error("FBO not complete");
      this.fbo[k] = fb;
    }
    this.tex.bmask = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex.bmask);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, NX, NY, 0, gl.RED, gl.UNSIGNED_BYTE, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  _destroyTextures() {
    if (!this.tex) return;
    const gl = this.gl;
    for (const t of Object.values(this.tex)) gl.deleteTexture(t);
    for (const f of Object.values(this.fbo)) gl.deleteFramebuffer(f);
    this.tex = null; this.fbo = null;
  }

  _uploadBoundaryMask() {
    const gl = this.gl;
    const NX = this.NX_full, NY = this.NY_full;
    const data = new Uint8Array(NX * NY);
    // The inlet/outlet fractions are taken over the interior height
    // (ny cells, j ∈ [1, NY-2]); j=0 / j=NY-1 are top/bottom walls.
    const jInL = Math.round(this.inletFracY[0] * this.ny) + 1;
    const jInR = Math.round(this.inletFracY[1] * this.ny) + 1;
    const jOutL = Math.round(this.outletFracY[0] * this.ny) + 1;
    const jOutR = Math.round(this.outletFracY[1] * this.ny) + 1;
    for (let j = 0; j < NY; j++) {
      for (let i = 0; i < NX; i++) {
        let bt = 0;
        if (i === 0) {
          bt = (j >= jInL && j < jInR) ? 5 : 1;
        } else if (i === NX - 1) {
          bt = (j >= jOutL && j < jOutR) ? 6 : 2;
        } else if (j === 0) {
          bt = 3;
        } else if (j === NY - 1) {
          bt = 4;
        }
        data[j * NX + i] = bt;
      }
    }
    gl.bindTexture(gl.TEXTURE_2D, this.tex.bmask);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, NX, NY, 0, gl.RED, gl.UNSIGNED_BYTE, data);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  }

  _fillInitialState(h0) {
    const gl = this.gl;
    const NX = this.NX_full, NY = this.NY_full;
    const data = new Float32Array(NX * NY * 4);
    for (let k = 0; k < NX * NY; k++) data[k * 4] = h0;
    gl.bindTexture(gl.TEXTURE_2D, this.tex.A);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, NX, NY, gl.RGBA, gl.FLOAT, data);
  }

  setMode(m) { this.mode = m; }
  setScale(s) { this.scale = Math.max(1e-6, s); }
  setLUT(bytes) {
    const gl = this.gl;
    if (!this.lutTex) {
      this.lutTex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    } else gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, bytes);
  }

  setQCurve(points) {
    if (Array.isArray(points) && points.length >= 1) {
      this.qCurveData = points.slice();
    }
    this._applyQCurve();
  }

  _applyQCurve() {
    if (!this.params) return;
    const tb = this.meta.timelineBlock;
    const pts = this.qCurveData;
    for (let i = 0; i < tb.nPoints; i++) {
      this.params[tb.valueStart + i] =
        this._lerpCurve(pts, this.params[tb.timeStart + i]);
    }
  }

  _lerpCurve(pts, t) {
    if (pts.length === 1) return pts[0][1];
    if (t <= pts[0][0]) return pts[0][1];
    const last = pts[pts.length - 1];
    if (t >= last[0]) return last[1];
    for (let i = 0; i < pts.length - 1; i++) {
      const [t0, v0] = pts[i], [t1, v1] = pts[i + 1];
      if (t >= t0 && t <= t1) {
        const u = t1 > t0 ? (t - t0) / (t1 - t0) : 0;
        return v0 * (1 - u) + v1 * u;
      }
    }
    return last[1];
  }

  _step() {
    const gl = this.gl;
    const NX = this.NX_full, NY = this.NY_full;
    gl.useProgram(this.simProg);
    gl.bindVertexArray(this.vao);
    gl.viewport(0, 0, NX, NY);
    const loc = n => this._uloc(this.simProg, n);
    gl.uniform1i(loc("uA"), 0);
    gl.uniform1i(loc("uB"), 1);
    gl.uniform1i(loc("uBmask"), 2);
    gl.uniform1f(loc("uDt"), this.dt);
    gl.uniform1f(loc("uDx"), this.dx);
    gl.uniform1fv(loc("uP"), this.params);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.tex.bmask);
    // Stage 0: A → B
    gl.uniform1f(loc("uTime"), this.time);
    gl.uniform1i(loc("uStage"), 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo.B);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.tex.A);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    // Stage 1: A (q^n) + B (q*) → C
    gl.uniform1f(loc("uTime"), this.time + this.dt);
    gl.uniform1i(loc("uStage"), 1);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo.C);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.tex.B);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.tex.A);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    // Swap A ↔ C
    const tT = this.tex.A; this.tex.A = this.tex.C; this.tex.C = tT;
    const fT = this.fbo.A; this.fbo.A = this.fbo.C; this.fbo.C = fT;
    this.time += this.dt;
  }

  /** Advance until ``this.time >= tTarget`` (uses the fixed dt). */
  stepTo(tTarget) {
    while (this.time + 1e-9 < tTarget) {
      this._step();
      if (this.time >= this.endTime) return;
    }
  }

  render() {
    const gl = this.gl;
    if (!this.lutTex) return;
    const modeIdx = this.mode === "vmag" ? 1 : 0;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.renderProg);
    gl.bindVertexArray(this.vao);
    const loc = n => this._uloc(this.renderProg, n);
    gl.uniform1i(loc("uA"), 0);
    gl.uniform1i(loc("uLUT"), 1);
    gl.uniform1i(loc("uMode"), modeIdx);
    gl.uniform1i(loc("uNY"), this.ny);
    gl.uniform1f(loc("uScale"), 254.0 / Math.max(1e-6, this.scale));
    gl.uniform1f(loc("uWet"), this.wet);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.tex.A);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.lutTex);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
  }

  /**
   * Read the interior state (h, hu, hv) back from the GPU as a
   * Float32Array of length ``nx * ny * 3`` in row-major order with
   * ``j`` increasing south → north (i.e. j=0 is the first row of the
   * output, matching the WebGL framebuffer y origin). This is the
   * canonical raw field the postprocess script colours.
   */
  readInteriorRGB() {
    const gl = this.gl;
    const NX = this.NX_full, NY = this.NY_full;
    const buf = new Float32Array(NX * NY * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo.A);
    gl.readPixels(0, 0, NX, NY, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const out = new Float32Array(this.nx * this.ny * 3);
    for (let j = 0; j < this.ny; j++) {
      for (let i = 0; i < this.nx; i++) {
        const src = ((j + 1) * NX + (i + 1)) * 4;
        const dst = (j * this.nx + i) * 3;
        out[dst]     = buf[src];
        out[dst + 1] = buf[src + 1];
        out[dst + 2] = buf[src + 2];
      }
    }
    return out;
  }

  /** Readback the interior h field — for auto-scaling the live preview. */
  measureMaxField() {
    const gl = this.gl;
    const NX = this.NX_full, NY = this.NY_full;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fbo.A);
    const buf = new Float32Array(NX * NY * 4);
    gl.readPixels(0, 0, NX, NY, gl.RGBA, gl.FLOAT, buf);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    let mh = 0, mv = 0;
    for (let j = 1; j < NY - 1; j++) {
      for (let i = 1; i < NX - 1; i++) {
        const k = j * NX + i;
        const h = buf[k * 4];
        if (h > mh) mh = h;
        if (h > this.wet) {
          const u = buf[k * 4 + 1] / h, v = buf[k * 4 + 2] / h;
          const vm = Math.sqrt(u * u + v * v);
          if (vm > mv) mv = vm;
        }
      }
    }
    return { maxH: mh, maxV: mv };
  }
}
