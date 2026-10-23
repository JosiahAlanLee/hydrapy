"""Physarum: slime-mould agents that lay trails and follow them (Jeff Jones, 2010).

Tens of thousands of agents wander a grid. Each step an agent smells the trail at three points
ahead of it (left, centre, right), turns towards the strongest, moves one cell and deposits a
little trail; the trail spreads and fades.  Networks form, thicken where traffic is heavy, and
re-route — the transport networks of *Physarum polycephalum*, and the look Sage Jenson made famous.

    mould = Physarum(size=(320, 180), agents=(256, 128))
    src(mould).out(o0)

    voronoi(4, 0.2).out(o1)
    src(Physarum(food=o1, food_weight=2.0)).out(o0)      # the agents follow the picture as well as each other

``sensor_angle``, ``sensor_distance``, ``turn``, ``step``, ``deposit``, ``decay`` take signals, so a
modular can retune the network live (small sensor angles: long straight veins; large: fine mesh).
Runs on the NumPy and GL backends and in the HTML export (WebGL2).
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .core import Output, Texture
from .noise import h21
from .signals import as_signal

_PARAMS = ("sensor_angle", "sensor_distance", "turn", "step", "deposit", "decay", "diffuse", "food_weight", "exposure")


class Physarum(Texture):
    _counter = 0

    def __init__(self, size=(320, 180), agents=(256, 128), sensor_angle=0.6, sensor_distance=20.0, turn=0.4,
                 step=1.0, deposit=0.04, decay=0.93, diffuse=0.6, food: Optional[Output] = None, food_weight=1.0,
                 exposure=0.5, colour=(1.0, 0.85, 0.55), speed: float = 60.0, max_steps: int = 4, seed: float = 0.0):
        self.size = (int(size[0]), int(size[1]))
        self.agents = (int(agents[0]), int(agents[1]))
        vals = dict(sensor_angle=sensor_angle, sensor_distance=sensor_distance, turn=turn, step=step, deposit=deposit,
                    decay=decay, diffuse=diffuse, food_weight=food_weight, exposure=exposure)
        self.params = {k: as_signal(v) for k, v in vals.items()}
        self.food, self.colour = food, tuple(float(c) for c in colour)
        self.speed, self.max_steps, self.seed = float(speed), int(max_steps), float(seed)
        Physarum._counter += 1
        self.name = f"physarum{Physarum._counter}"
        self.reset()

    @property
    def count(self) -> int:
        return self.agents[0] * self.agents[1]

    def reset(self):
        self.state = None            # (N, 4) x, y, heading, 0 — built lazily (NumPy) or on the GPU
        self.trail = np.zeros((self.size[1], self.size[0]), np.float32)
        self.step_index = 0
        self._target = None
        self.gpu_reset = True
        self._visible = None

    def initial_state(self) -> np.ndarray:
        """Agents at hashed positions and headings (the same numbers the GPU init pass computes)."""
        aw, ah = self.agents
        iy, ix = np.mgrid[0:ah, 0:aw].astype(np.float32)
        i = (iy * np.float32(aw) + ix).reshape(-1)
        s = np.float32(self.seed) * np.float32(131.0)
        x = h21(np.stack([i, np.full_like(i, s + np.float32(1.0))], -1)) * np.float32(self.size[0])
        y = h21(np.stack([i, np.full_like(i, s + np.float32(2.0))], -1)) * np.float32(self.size[1])
        a = h21(np.stack([i, np.full_like(i, s + np.float32(3.0))], -1)) * np.float32(6.283185307)
        return np.stack([x, y, a, np.zeros_like(a)], -1).astype(np.float32)

    def steps_due(self, t: float) -> int:
        target = math.floor(t * self.speed + 1e-9)
        if self._target is None or target < self._target:
            if self._target is not None:
                self.reset()
            self._target = target
            return 1 if self.step_index == 0 else 0
        n = min(target - self._target, self.max_steps)
        self._target = target
        return n

    def values(self, ctx) -> dict:
        return {k: float(s(ctx)) for k, s in self.params.items()}

    # -- NumPy ------------------------------------------------------------------
    def step_np(self, p: dict, food: Optional[np.ndarray] = None):
        if self.state is None:
            self.state = self.initial_state()
        W, H = self.size
        x, y, a = self.state[:, 0], self.state[:, 1], self.state[:, 2]

        def smell(ang):
            sx = np.mod(x + np.cos(ang) * p["sensor_distance"], W)
            sy = np.mod(y + np.sin(ang) * p["sensor_distance"], H)
            ix, iy = sx.astype(int) % W, sy.astype(int) % H
            v = self.trail[iy, ix]
            if food is not None:
                v = v + p["food_weight"] * food[iy, ix]
            return v

        sa, ta = np.float32(p["sensor_angle"]), np.float32(p["turn"])
        l, c, r = smell(a + sa), smell(a), smell(a - sa)
        n = self.count
        idx = np.arange(n, dtype=np.float32)
        coin = h21(np.stack([idx, np.full_like(idx, np.float32(self.step_index) + np.float32(self.seed) * np.float32(131.0) + np.float32(7.0))], -1))
        rnd = np.where(coin < 0.5, ta, -ta)
        da = np.where((c > l) & (c > r), 0.0, np.where((c < l) & (c < r), rnd, np.where(l > r, ta, -ta)))
        a = a + da.astype(np.float32)
        x = np.mod(x + np.cos(a) * p["step"], W).astype(np.float32)
        y = np.mod(y + np.sin(a) * p["step"], H).astype(np.float32)
        self.state = np.stack([x, y, a, self.state[:, 3]], -1).astype(np.float32)
        np.add.at(self.trail, (y.astype(int) % H, x.astype(int) % W), np.float32(p["deposit"]))
        t = self.trail
        blur = sum(np.roll(np.roll(t, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / np.float32(9.0)
        self.trail = ((t * (1 - p["diffuse"]) + blur * p["diffuse"]) * p["decay"]).astype(np.float32)
        self.step_index += 1

    def advance_np(self, env, t: float) -> np.ndarray:
        n = self.steps_due(t)
        if n:
            p = self.values(env.ctx)
            food = None
            if self.food is not None:
                w, h = self.size
                ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
                f = env.sample(env.texture(self.food), np.stack([(xs + 0.5) / w, (ys + 0.5) / h], -1))
                food = f[..., 0] * 0.2125 + f[..., 1] * 0.7154 + f[..., 2] * 0.0721
            for _ in range(n):
                self.step_np(p, food)
            self._visible = None
        if self._visible is None:
            p = self.values(env.ctx)
            g = 1.0 - np.exp(-self.trail * p["exposure"])
            rgb = np.stack([g * self.colour[0], g * self.colour[1], g * self.colour[2]], -1)
            self._visible = np.concatenate([rgb, np.ones(g.shape + (1,))], -1).astype(np.float32)
        return self._visible

    # -- GLSL ---------------------------------------------------------------------
    def glsl(self) -> dict:
        """Shader sources (ES 1.00 fragment shaders except ``deposit_vert``/``deposit_frag``, ES 3.00 / 330)."""
        W, H = self.size
        aw, ah = self.agents
        f = lambda v: f"{float(v):.7g}" + ("" if "." in f"{float(v):.7g}" or "e" in f"{float(v):.7g}" else ".0")
        init = f"""uniform float seed;
void main() {{
  vec2 cell = floor(gl_FragCoord.xy);
  float i = cell.y * {f(aw)} + cell.x;
  float s = seed * 131.0;
  gl_FragColor = vec4(_h21(vec2(i, s + 1.0)) * {f(W)}, _h21(vec2(i, s + 2.0)) * {f(H)}, _h21(vec2(i, s + 3.0)) * 6.283185307, 0.0);
}}
"""
        update = f"""uniform sampler2D agents, trail, food;
uniform float stepIndex, seed, sensorAngle, sensorDistance, turn, stepSize, foodWeight, foodOn;
const vec2 GRID = vec2({f(W)}, {f(H)});
float smell(vec2 p, float a) {{
  vec2 s = mod(p + vec2(cos(a), sin(a)) * sensorDistance, GRID);
  vec2 uv = (floor(s) + 0.5) / GRID;
  float v = texture2D(trail, uv).r;
  if (foodOn > 0.5) v += foodWeight * _luminance(texture2D(food, uv).rgb);
  return v;
}}
void main() {{
  vec2 cell = floor(gl_FragCoord.xy);
  vec4 ag = texture2D(agents, (cell + 0.5) / vec2({f(aw)}, {f(ah)}));
  float a = ag.z;
  float l = smell(ag.xy, a + sensorAngle), c = smell(ag.xy, a), r = smell(ag.xy, a - sensorAngle);
  float i = cell.y * {f(aw)} + cell.x;
  float coin = _h21(vec2(i, stepIndex + seed * 131.0 + 7.0));
  float da = (c > l && c > r) ? 0.0 : ((c < l && c < r) ? (coin < 0.5 ? turn : -turn) : (l > r ? turn : -turn));
  a += da;
  vec2 p = mod(ag.xy + vec2(cos(a), sin(a)) * stepSize, GRID);
  gl_FragColor = vec4(p, a, ag.w);
}}
"""
        diffuse = f"""uniform sampler2D trail;
uniform float decay, diffuse;
void main() {{
  vec2 px = 1.0 / vec2({f(W)}, {f(H)});
  vec2 uv = gl_FragCoord.xy * px;
  float c = texture2D(trail, uv).r, s = 0.0;
  for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++)
    s += texture2D(trail, fract(uv + vec2(float(dx), float(dy)) * px)).r;
  float v = (c * (1.0 - diffuse) + s / 9.0 * diffuse) * decay;
  gl_FragColor = vec4(v, 0.0, 0.0, 1.0);
}}
"""
        view = f"""uniform sampler2D trail;
uniform float exposure;
void main() {{
  float g = 1.0 - exp(-texture2D(trail, gl_FragCoord.xy / vec2({f(W)}, {f(H)})).r * exposure);
  gl_FragColor = vec4(vec3({f(self.colour[0])}, {f(self.colour[1])}, {f(self.colour[2])}) * g, 1.0);
}}
"""
        # points: one vertex per agent, reading its position from the agent texture
        deposit_vert = f"""uniform sampler2D agents;
void main() {{
  int i = gl_VertexID;
  vec4 ag = texelFetch(agents, ivec2(i % {aw}, i / {aw}), 0);
  vec2 cell = floor(ag.xy) + 0.5;
  gl_Position = vec4(cell / vec2({f(W)}, {f(H)}) * 2.0 - 1.0, 0.0, 1.0);
  gl_PointSize = 1.0;
}}
"""
        deposit_frag = """uniform float amount;
OUT_DECL
void main() { OUT_VAR = vec4(amount, 0.0, 0.0, 0.0); }
"""
        return {"init": init, "update": update, "diffuse": diffuse, "view": view,
                "deposit_vert": deposit_vert, "deposit_frag": deposit_frag}

    def spec(self, precision: str = "highp") -> dict:
        from .nca import shader_header
        g = self.glsl()
        head = shader_header(precision)
        return {"name": self.name, "w": self.size[0], "h": self.size[1], "aw": self.agents[0], "ah": self.agents[1],
                "speed": self.speed, "maxSteps": self.max_steps, "seed": self.seed,
                "init": head + g["init"], "update": head + g["update"], "diffuse": head + g["diffuse"],
                "view": head + g["view"],
                "depositVert": "#version 300 es\nprecision highp float;\nprecision highp int;\nprecision highp sampler2D;\n" + g["deposit_vert"],
                "depositFrag": "#version 300 es\nprecision highp float;\n" + g["deposit_frag"].replace("OUT_DECL", "out vec4 o;").replace("OUT_VAR", "o"),
                "food": self.food.name if self.food is not None else None,
                "params": {k: s.js() for k, s in self.params.items()}}

    # -- GL backend (moderngl) ----------------------------------------------------------
    def gl_resources(self, r):
        if getattr(self, "_gl", None) is not None and self._gl["renderer"] is r:
            return self._gl
        from .glsl import VERTEX_DESKTOP, to_desktop
        from .nca import shader_header
        g = self.glsl()
        head = shader_header(r.precision)
        ctx = r.ctx

        def prog(src):
            p = ctx.program(vertex_shader=VERTEX_DESKTOP, fragment_shader=to_desktop(head + src))
            return p, ctx.vertex_array(p, [(r.vbo, "2f", "position")])

        def target(size):
            t = ctx.texture(size, 4, dtype="f4")
            t.filter = (r.gl.NEAREST, r.gl.NEAREST)
            t.repeat_x = t.repeat_y = True
            fb = ctx.framebuffer(color_attachments=[t])
            fb.use(); ctx.clear(0.0, 0.0, 0.0, 0.0)
            return {"tex": t, "fbo": fb}

        dep = ctx.program(vertex_shader="#version 330 core\n" + g["deposit_vert"],
                          fragment_shader="#version 330 core\n" + g["deposit_frag"].replace("OUT_DECL", "out vec4 o;").replace("OUT_VAR", "o"))
        view = target(self.size)
        view["tex"].filter = (r.gl.LINEAR, r.gl.LINEAR)
        self._gl = {"renderer": r, "init": prog(g["init"]), "update": prog(g["update"]), "diffuse": prog(g["diffuse"]),
                    "view_prog": prog(g["view"]), "deposit": (dep, ctx.vertex_array(dep, [])),
                    "agents": [target(self.agents), target(self.agents)], "trail": [target(self.size), target(self.size)],
                    "view": view, "a": 0, "t": 0}
        return self._gl

    def gl_release(self, r):
        g = getattr(self, "_gl", None)
        if g is None or g["renderer"] is not r:
            return
        for key in ("init", "update", "diffuse", "view_prog", "deposit"):
            p, vao = g[key]
            vao.release(); p.release()
        for fb in g["agents"] + g["trail"] + [g["view"]]:
            fb["fbo"].release(); fb["tex"].release()
        self._gl = None

    def gl_step(self, r, ctx):
        g = self.gl_resources(r)
        n = self.steps_due(ctx.time)
        gl = r.ctx
        if self.gpu_reset:
            for fb in g["trail"]:
                fb["fbo"].use(); gl.clear(0.0, 0.0, 0.0, 0.0)
            p, vao = g["init"]
            g["agents"][g["a"]]["fbo"].use()
            r._set(p, "seed", self.seed)
            vao.render(r.gl.TRIANGLES)
            self.gpu_reset = False
        pv = self.values(ctx)
        food = r.outs[self.food.name]["ring"][0]["tex"] if self.food is not None else r._blank
        for _ in range(n):
            # 1. agents sense the trail, turn and move
            p, vao = g["update"]
            g["agents"][1 - g["a"]]["fbo"].use()
            g["agents"][g["a"]]["tex"].use(location=0); r._set(p, "agents", 0)
            g["trail"][g["t"]]["tex"].use(location=1); r._set(p, "trail", 1)
            food.use(location=2); r._set(p, "food", 2)
            for k, v in (("stepIndex", float(self.step_index)), ("seed", self.seed), ("sensorAngle", pv["sensor_angle"]),
                         ("sensorDistance", pv["sensor_distance"]), ("turn", pv["turn"]), ("stepSize", pv["step"]),
                         ("foodWeight", pv["food_weight"]), ("foodOn", 1.0 if self.food is not None else 0.0)):
                r._set(p, k, v)
            vao.render(r.gl.TRIANGLES)
            g["a"] = 1 - g["a"]
            # 2. they deposit (one point each, added)
            p, vao = g["deposit"]
            g["trail"][g["t"]]["fbo"].use()
            g["agents"][g["a"]]["tex"].use(location=0); r._set(p, "agents", 0)
            r._set(p, "amount", pv["deposit"])
            gl.enable(r.gl.BLEND); gl.blend_func = (r.gl.ONE, r.gl.ONE)
            vao.render(r.gl.POINTS, vertices=self.count)
            gl.disable(r.gl.BLEND)
            # 3. the trail spreads and fades
            p, vao = g["diffuse"]
            g["trail"][1 - g["t"]]["fbo"].use()
            g["trail"][g["t"]]["tex"].use(location=0); r._set(p, "trail", 0)
            r._set(p, "decay", pv["decay"]); r._set(p, "diffuse", pv["diffuse"])
            vao.render(r.gl.TRIANGLES)
            g["t"] = 1 - g["t"]
            self.step_index += 1
        p, vao = g["view_prog"]
        g["view"]["fbo"].use()
        g["trail"][g["t"]]["tex"].use(location=0); r._set(p, "trail", 0)
        r._set(p, "exposure", pv["exposure"])
        vao.render(r.gl.TRIANGLES)

    def gl_texture(self, r):
        return self.gl_resources(r)["view"]["tex"]

    def gl_state(self, r):
        """(agents (N, 4), trail (h, w)) read back from the GPU."""
        g = self.gl_resources(r)
        a = np.frombuffer(g["agents"][g["a"]]["tex"].read(), np.float32).reshape(-1, 4)
        t = np.frombuffer(g["trail"][g["t"]]["tex"].read(), np.float32).reshape(self.size[1], self.size[0], 4)[..., 0]
        return a, t
