"""Experimental desktop kernels; portable fragment shaders remain the reference.

One dispatch performs one iteration, writing the same RGBA16F target as a draw.
Workgroup barriers cannot synchronize the whole image, so iterations must never
be folded into an in-place loop. No CPU readback or extra texture is required.
"""
from .glsl import ComputeStep
from .noise import GLSL_NOISE


GROUP_SIZE = 16


_RD = """
vec2 ab(ivec2 q) {
    vec2 v = tile[q.y * TILE_SIZE + q.x].rg;
    if (v.x + v.y < 0.001) v.x = 1.0;
    return v;
}
vec4 update(ivec2 q, vec2 st) {
    vec2 cur = ab(q);
    float A = cur.x, B = cur.y;
    float lapA = -A, lapB = -B;
    for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
        if (i == 0 && j == 0) continue;
        float w = (i == 0 || j == 0) ? 0.2 : 0.05;
        vec2 v = ab(q + ivec2(i, j));
        lapA += w * v.x; lapB += w * v.y;
    }
    float abb = A * B * B;
    float A2 = A + (da * lapA - abb + feed * (1.0 - A));
    float B2 = B + (db * lapB + abb - (kill + feed) * B);
    return vec4(clamp(A2, 0.0, 1.0), clamp(B2, 0.0, 1.0), clamp(B2, 0.0, 1.0), 1.0);
}
"""


_BLEED = """
vec4 sampleTile(vec2 p) {
    ivec2 lo = ivec2(floor(p));
    vec2 f = fract(p);
    int i = lo.y * TILE_SIZE + lo.x;
    return mix(mix(tile[i], tile[i + 1], f.x),
               mix(tile[i + TILE_SIZE], tile[i + TILE_SIZE + 1], f.x), f.y);
}
vec4 update(ivec2 q, vec2 st) {
    vec4 acc = vec4(0.0); float wsum = 0.0;
    for (int y = -1; y <= 1; y++) for (int x = -1; x <= 1; x++) {
        float w = (x == 0 && y == 0) ? 2.0 : 1.0;
        // Interpolate straight RGBA first, then premultiply, as in the fragment op.
        vec4 s = sampleTile(vec2(q) + vec2(x, y) * 1.5);
        acc += vec4(s.rgb * s.a, s.a) * w; wsum += w;
    }
    acc /= wsum;
    float n = _noise(vec3(st * resolution * 0.04, 11.0));
    float a = smoothstep(0.5 - grow - softness, 0.5 - grow + softness,
                         acc.a + (n - 0.5) * roughness) * max(acc.a, 1.0);
    vec3 rgb = acc.a > 1e-4 ? acc.rgb / acc.a : vec3(1.0);
    return vec4(rgb, a);
}
"""


def source(step: ComputeStep, uniforms) -> str:
    halo, body = {"rd": (1, _RD), "bleed": (2, _BLEED)}[step.kernel]
    tile_size = GROUP_SIZE + 2 * halo
    declarations = "\n".join(f"uniform float {u.name};" for u in uniforms)
    arguments = "\n".join(f"#define {name} ({value})" for name, value in step.arguments.items())
    noise = GLSL_NOISE if step.kernel == "bleed" else ""
    return f"""#version 430 core
layout(local_size_x = {GROUP_SIZE}, local_size_y = {GROUP_SIZE}) in;
layout(rgba16f, binding = 0) uniform writeonly image2D destination;
uniform sampler2D {step.sampler};
uniform vec2 resolution;
{declarations}
#define TILE_SIZE {tile_size}
shared vec4 tile[TILE_SIZE * TILE_SIZE];
{noise}
{arguments}
{body}
void main() {{
    ivec2 extent = ivec2(resolution);
    ivec2 origin = ivec2(gl_WorkGroupID.xy) * {GROUP_SIZE} - {halo};
    // Every invocation loads and reaches the barrier, even in a partial group.
    for (int i = int(gl_LocalInvocationIndex); i < TILE_SIZE * TILE_SIZE; i += {GROUP_SIZE**2}) {{
        ivec2 p = origin + ivec2(i % TILE_SIZE, i / TILE_SIZE);
        // GLSL integer remainder is undefined for negative operands. Add enough
        // periods first, including for a one-pixel texture with a two-pixel halo.
        p = (p + extent * {halo}) % extent;
        tile[i] = texelFetch({step.sampler}, p, 0);
    }}
    barrier();
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(pixel, extent))) return;
    ivec2 q = ivec2(gl_LocalInvocationID.xy) + {halo};
    vec2 st = (vec2(pixel) + 0.5) / resolution;
    imageStore(destination, pixel, update(q, st));
}}
"""
