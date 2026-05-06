// UI registry. This is like a mini web-framework so that i don't need to do a bunch of the pattern of
// `let variable_name = document.getElementById("id_name")` or something
const ui = new Map();
document.querySelectorAll("[data]").forEach((el) => {
  if (!el.id) return console.warn("data element missing id", el);
  ui.set(el.id, el);
});

// I only ended up using this like once, but the was to be able
// to make elements programatically and still have them in the ui registry
function create_data_element(tag, id) {
  const el = document.createElement(tag);
  el.id = id;
  el.setAttribute("data", "");
  ui.set(id, el);
  return el;
}

// Spline activation curve.
// This uses a lookup table. Then the compute-shader just lerps between the values
// This is a lot faster than having to do spline computation for every pixel

const LUT_SIZE = 256;
const LUT_MIN = -4.0;
const LUT_MAX = +4.0;

// I previously had the functions that used `cs` be pure functions, but I have decided that is in fact
// easier for me to not do that.
const canvas_spliner = new CanvasSpliner.CanvasSpliner(
  "spliner_parent",
  300,
  300,
);
// Canvas spliner is the size of the canvas

function init_activation_random() {
  for (let i = 0; i <= 1; i += 0.2) {
    canvas_spliner.add({ x: i, y: Math.random() });
  }
}

init_activation_random();
function build_lut() {
  // Pre-allocating a buffer lets me pretend I'm working at lower level
  const lut = new Float32Array(LUT_SIZE);
  for (let i = 0; i < LUT_SIZE; i++) {
    const t = i / (LUT_SIZE - 1);
    lut[i] = LUT_MIN + canvas_spliner.getValue(t) * (LUT_MAX - LUT_MIN);
  }
  return lut;
}

// Kernel Inputs
const k_inputs = [];
for (let i = 0; i < 9; i++) {
  const inp = create_data_element("input", `k${i}`);
  inp.type = "number";
  inp.step = "0.01";
  inp.value = (Math.random() * 2 - 1).toFixed(3);
  ui.get("kernel_grid").appendChild(inp);
  k_inputs.push(inp);
}

function get_kernel() {
  return k_inputs.map((inp) => Number(inp.value) || 0);
}
function set_kernel(v) {
  v.forEach((x, i) => k_inputs[i].value = x.toFixed(3));
}

// It is kinda funny that webgpu abstracs away swapchains and I'm now kinda re-doing that, but b/c this is a CA, that is in fact necessary
function make_ping_pong_bind_groups(layout, tex_a, tex_b, res_fn) {
  return {
    a: device.createBindGroup({ layout, entries: res_fn(tex_a, tex_b) }),
    b: device.createBindGroup({ layout, entries: res_fn(tex_b, tex_a) }),
  };
}

function make_texture_pair(n) {
  const desc = {
    size: [n, n],
    format: "r32float",
    usage: GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_DST,
  };
  return { a: device.createTexture(desc), b: device.createTexture(desc) };
}

function make_buffer(usage, data) {
  const buf = device.createBuffer({
    size: data.byteLength,
    usage: usage | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(buf, 0, data);
  return buf;
}

// Remembered that you can define a bunch of JS variables with only one let statement
let device,
  context,
  gpu_format,
  compute_pipeline,
  render_pipeline,
  compute_bind_layout,
  render_bind_layout,
  textures,
  compute_groups,
  render_groups,
  param_buf,
  lut_buf,
  color_buf,
  ping_a = true,
  frame_count = 0,
  last_fps_time = performance.now(),
  // Size defaults to 512.
  size = 512;

const COMPUTE_SHADER = `
struct Params {
  k0:f32, k1:f32, k2:f32,
  k3:f32, k4:f32, k5:f32,
  k6:f32, k7:f32, k8:f32,
  N:u32,
  // _p0:f32,_p1:f32,_p2:f32,_p3:f32,_p4:f32,_p5:f32,
}

const LUT_SIZE : u32 = 256u;
const LUT_MIN  : f32 = -4.0;
const LUT_MAX  : f32 =  4.0;

@group(0) @binding(0) var src                : texture_2d<f32>;
@group(0) @binding(1) var dst                : texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> p         : Params;
@group(0) @binding(3) var<storage, read> lut : array<f32, 256>;

fn activate(x: f32) -> f32 {
  let t  = clamp((x - LUT_MIN) / (LUT_MAX - LUT_MIN), 0.0, 1.0);
  // fi, hi, lo, fum (lut)
  // fi: float_index
  // hi: upper integer index
  // lo: lower integer index
  let fi = t * f32(LUT_SIZE - 1u);
  let lo = u32(fi);
  let hi = min(lo + 1u, LUT_SIZE - 1u);
  // Straight up lerping it, and by it, haha, well spline.
  return mix(lut[lo], lut[hi], fi - f32(lo));
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let n  = p.N;
  if gid.x >= n || gid.y >= n { return; }

  let x  = i32(gid.x);
  let y  = i32(gid.y);
  let iN = i32(n);

  var conv = 0.0;
  var kernel_idx   = 0u;

  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let nx = (x + dx + iN) % iN;
      let ny = (y + dy + iN) % iN;
      let s  = textureLoad(src, vec2<i32>(nx, ny), 0).r;
      let kv = array<f32, 9>(
        p.k0, p.k1, p.k2,
        p.k3, p.k4, p.k5,
        p.k6, p.k7, p.k8
      )[kernel_idx];
      conv += kv * s;
      kernel_idx++;
    }
  }

  let out = clamp(activate(conv), -1.0, 1.0);
  textureStore(dst, vec2<i32>(x, y), vec4<f32>(out, 0.0, 0.0, 1.0));
}
`;

const RENDER_SHADER = `
// Single hue; positive values are bright, negative values are dim.
@group(0) @binding(0) var state : texture_2d<f32>;
@group(0) @binding(1) var samp  : sampler;
@group(0) @binding(2) var<uniform> hue : vec4<f32>;  // rgb + pad

struct VO {
  @builtin(position) pos : vec4<f32>,
  @location(0)       uv  : vec2<f32>,
};

// TODO: perhaps change
@vertex fn vs(@builtin(vertex_index) vi: u32) -> VO {
  var P = array<vec2<f32>, 4>(
    vec2(-1.0, -1.0), vec2(1.0, -1.0),
    vec2(-1.0,  1.0), vec2(1.0,  1.0)
  );
  var U = array<vec2<f32>, 4>(
    vec2(0.0, 1.0), vec2(1.0, 1.0),
    vec2(0.0, 0.0), vec2(1.0, 0.0)
  );
  var o: VO;
  o.pos = vec4<f32>(P[vi], 0., 1.);
  o.uv  = U[vi];
  return o;
}

@fragment fn fs(i: VO) -> @location(0) vec4<f32> {
  let cell_val   = textureSample(state, samp, i.uv).r;
  let mag = abs(cell_val);
  var col : vec3<f32>;
  if cell_val >= 0.0 {
    // brightness is set to magnitude
    // Positive: full hue,
    col = hue.rgb * mag;
  } else {
    // Negative: desaturate toward grey and dim
    let luma  = dot(hue.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let desat = mix(hue.rgb, vec3<f32>(luma, luma, luma), 0.85);
    col = desat * mag * 0.4;
  }
  return vec4<f32>(col, 1.0);
}
`;

// Initialization and boring setup with webpug
async function init_webgpu() {
  if (!navigator.gpu) return false;
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) return false;
  device = await adapter.requestDevice();
  gpu_format = navigator.gpu.getPreferredCanvasFormat();
  context = ui.get("ca_canvas").getContext("webgpu");
  context.configure({ device, format: gpu_format, alphaMode: "opaque" });
  return true;
}

function build_pipelines() {
  const compute_mod = device.createShaderModule({ code: COMPUTE_SHADER });
  const render_mod = device.createShaderModule({ code: RENDER_SHADER });

  compute_bind_layout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        texture: { sampleType: "unfilterable-float" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        storageTexture: { access: "write-only", format: "r32float" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
      {
        binding: 3,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "read-only-storage" },
      },
    ],
  });

  compute_pipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [compute_bind_layout],
    }),
    compute: { module: compute_mod, entryPoint: "main" },
  });

  render_bind_layout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.FRAGMENT,
        texture: { sampleType: "unfilterable-float" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.FRAGMENT,
        sampler: { type: "non-filtering" },
      },
      {
        binding: 2,
        visibility: GPUShaderStage.FRAGMENT,
        buffer: { type: "uniform" },
      },
    ],
  });

  render_pipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [render_bind_layout],
    }),
    vertex: { module: render_mod, entryPoint: "vs" },
    fragment: {
      module: render_mod,
      entryPoint: "fs",
      targets: [{ format: gpu_format }],
    },
    primitive: { topology: "triangle-strip" },
  });
}

function make_param_data() {
  const k = get_kernel();
  const data = new Float32Array(16);
  for (let i = 0; i < 9; i++) data[i] = k[i];
  // This is kinda shitty, but it lets me just put everything in one buffer
  (new Uint32Array(data.buffer))[9] = size;
  return data;
}

// const pipe = (...fns) => x => fns.reduce((v, f) => f(v), x);
const niladic_pipe = (...fns) => (x) => fns.reduce((v, f) => f(v), x);

function rebuild_gpu_resources() {
  // Foolish soul: "I love using a GC'd language! It means I don't need to think about resource lifetimes"
  // The humble:
  if (textures) {
    textures.a.destroy();
    textures.b.destroy();
  }
  if (param_buf) param_buf.destroy();
  if (lut_buf) lut_buf.destroy();
  if (color_buf) color_buf.destroy();

  textures = make_texture_pair(size);
  param_buf = make_buffer(GPUBufferUsage.UNIFORM, make_param_data());
  lut_buf = make_buffer(GPUBufferUsage.STORAGE, build_lut());
  color_buf = make_buffer(
    GPUBufferUsage.UNIFORM,
    niladic_pipe(
      () => {
        const h = Math.random();
        const s = 0.7 + Math.random() * 0.3;
        // brightness caled by CA itself
        const l = 0.5;
        return [h, s, l];
      },
      ([h, s, l]) => {
        // Yoinked from https://gist.github.com/mjackson/5311256
        var r, g, b;

        if (s == 0) {
          r = g = b = l; // achromatic
        } else {
          function hue2rgb(p, q, t) {
            if (t < 0) t += 1;
            if (t > 1) t -= 1;
            if (t < 1 / 6) return p + (q - p) * 6 * t;
            if (t < 1 / 2) return q;
            if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
            return p;
          }

          var q = l < 0.5 ? l * (1 + s) : l + s - l * s;
          var p = 2 * l - q;

          r = hue2rgb(p, q, h + 1 / 3);
          g = hue2rgb(p, q, h);
          b = hue2rgb(p, q, h - 1 / 3);
        }

        return [r, g, b];
      },
      (rgb) => {
        // vec4 (rgb + pad) = 16 bytes
        const d = new Float32Array(4);
        d[0] = rgb[0];
        d[1] = rgb[1];
        d[2] = rgb[2];
        d[3] = 0;
        return d;
      },
    )(),
  );

  // More boring webgpu plumbing
  const samp = device.createSampler({
    magFilter: "nearest",
    minFilter: "nearest",
  });

  compute_groups = make_ping_pong_bind_groups(
    compute_bind_layout,
    textures.a,
    textures.b,
    (src, dst) => [
      { binding: 0, resource: src.createView() },
      { binding: 1, resource: dst.createView() },
      { binding: 2, resource: { buffer: param_buf } },
      { binding: 3, resource: { buffer: lut_buf } },
    ],
  );

  render_groups = make_ping_pong_bind_groups(
    render_bind_layout,
    textures.a,
    textures.b,
    (src, _dst) => [
      { binding: 0, resource: src.createView() },
      { binding: 1, resource: samp },
      { binding: 2, resource: { buffer: color_buf } },
    ],
  );
}

function seed_grid() {
  const data = new Float32Array(size * size);
  for (let i = 0; i < data.length; i++) data[i] = (Math.random() * 2 - 1) * 0.3;
  device.queue.writeTexture({ texture: textures.a }, data, {
    bytesPerRow: size * 4,
  }, { width: size, height: size });
  ping_a = true;
}

async function setup() {
  size = Number(ui.get("grid_size").value);
  const display_size = Math.min(
    window.innerWidth - 24,
    window.innerHeight - 280,
    780,
  );
  // I used `deno fmt` for the file. It's choice to indent like this befuddles me.
  ui.get("ca_canvas").width = ui.get("ca_canvas").height = size;
  ui.get("ca_canvas").style.width =
    ui.get("ca_canvas").style.height =
      display_size + "px";

  rebuild_gpu_resources();
  seed_grid();
}

let mouse_down = false;
ui.get("ca_canvas").addEventListener("mousedown", (e) => {
  mouse_down = true;
  draw_on(e);
});
ui.get("ca_canvas").addEventListener("mouseup", () => {
  mouse_down = false;
});
ui.get("ca_canvas").addEventListener("mousemove", (e) => {
  if (mouse_down) draw_on(e);
});

function draw_on(e) {
  const canvas_rect = ui.get("ca_canvas").getBoundingClientRect();
  cursor_x = Math.floor(
    (e.clientX - canvas_rect.left) / canvas_rect.width * size,
  ),
    cursor_y = Math.floor(
      (e.clientY - canvas_rect.top) / canvas_rect.height * size,
    ),
    brush_radius = 8,
    brush_diameter = brush_radius * 2,
    // We reshape it, but fundamentally this is just a square of full 1.0s. So fully alive
    patch = new Float32Array(brush_diameter * brush_diameter).fill(1.0),
    tile_x = Math.max(0, cursor_x - brush_radius),
    tile_y = Math.max(0, cursor_y - brush_radius),
    write_width = Math.min(brush_diameter, size - tile_x),
    write_height = Math.min(brush_diameter, size - tile_y);
  if (write_width <= 0 || write_height <= 0) return;
  // Blit the square on to image
  device.queue.writeTexture(
    {
      texture: ping_a ? textures.a : textures.b,
      origin: { x: tile_x, y: tile_y },
    },
    patch,
    { bytesPerRow: brush_diameter * 4 },
    { width: write_width, height: write_height },
  );
}

// Render loop
function frame() {
  if (!device) return;

  device.queue.writeBuffer(param_buf, 0, make_param_data());

  const steps = Number(ui.get("speed").value);
  const workgroup_size = Math.ceil(size / 16);
  const cmd_encoder = device.createCommandEncoder();

  for (let s = 0; s < steps; s++) {
    const compute_pass = cmd_encoder.beginComputePass();
    compute_pass.setPipeline(compute_pipeline);
    compute_pass.setBindGroup(0, ping_a ? compute_groups.a : compute_groups.b);
    compute_pass.dispatchWorkgroups(workgroup_size, workgroup_size);
    compute_pass.end();
    // Double buffering switch
    ping_a = !ping_a;
  }

  const view = context.getCurrentTexture().createView();
  const render_pass = cmd_encoder.beginRenderPass({
    colorAttachments: [{
      view,
      loadOp: "clear",
      clearValue: [0, 0, 0, 1],
      storeOp: "store",
    }],
  });
  render_pass.setPipeline(render_pipeline);
  render_pass.setBindGroup(0, ping_a ? render_groups.a : render_groups.b);
  render_pass.draw(4);
  render_pass.end();

  device.queue.submit([cmd_encoder.finish()]);

  frame_count++;
  const now = performance.now();
  if (now - last_fps_time > 600) {
    ui.get("fps").textContent =
      (frame_count / ((now - last_fps_time) / 1000)).toFixed(0) + " fps";
    frame_count = 0;
    last_fps_time = now;
  }

  requestAnimationFrame(frame);
}

// Boot the loop
(async () => {
  const ok = await init_webgpu();
  if (!ok) {
    const p = document.createElement("p");
    p.style.color = "red";
    p.textContent = "WebGPU not available.";
    document.body.insertAdjacentElement("afterbegin", p);
    return;
  }
  build_pipelines();
  await setup();
  frame();
})();

canvas_spliner.on("movePoint", () => {
  device.queue.writeBuffer(lut_buf, 0, build_lut());
});
