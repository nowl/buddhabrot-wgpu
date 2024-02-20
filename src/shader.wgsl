@group(0) @binding(0)
var<storage, read_write> counts: array<atomic<u32>>; // this is used as both input and output for convenience

@group(0) @binding(1)
var<storage, read> prng_data: array<f32>; // this is used as both input and output for convenience

struct GPUVars {
    width: u32,
    height: u32,
    max_iterations: u32,
    ll_re: f32,
    ll_im: f32,
    ur_re: f32,
    ur_im: f32,
    zoom_ll_re: f32,
    zoom_ll_im: f32,
    zoom_ur_re: f32,
    zoom_ur_im: f32,
}

@group(0) @binding(2)
var<uniform> vars_data: GPUVars;

fn world_to_screen(cr: f32, ci: f32) -> vec2f {
    let w = f32(vars_data.width);
    let h = f32(vars_data.height);
    let x = (cr - vars_data.zoom_ll_re) / (vars_data.zoom_ur_re - vars_data.zoom_ll_re) * w;
    let y = (ci - vars_data.zoom_ll_im) / (vars_data.zoom_ur_im - vars_data.zoom_ll_im) * h;
    return vec2f(x, y);
}

fn buddhabrot_iterations(p1: f32, p2: f32) {
    let re = p1 * (vars_data.ur_re - vars_data.ll_re) + vars_data.ll_re;
    let im = p2 * (vars_data.ur_im - vars_data.ll_im) + vars_data.ll_im;

    // check for escape
    var iters: u32 = 0u;
    var r: f32 = 0.0;
    var i: f32 = 0.0;
    var cr: f32 = re;
    var ci: f32 = im;
    loop {
        iters = iters + 1;
        if iters >= vars_data.max_iterations {
            return;
        }

        let tr = r;
        r = r*r - i*i + cr;
        i = 2*tr*i + ci;

        if r * r + i * i > 8.0 {
            break;
        }
    }

    // now redo with counts
    iters = 0u;
    r = 0.0;
    i = 0.0;
    cr = re;
    ci = im;
    loop {
        iters = iters + 1;

        let tr = r;
        r = r*r - i*i + cr;
        i = 2*tr*i + ci;

        if r * r + i * i > 8.0 {
            break;
        }

        let pos = world_to_screen(r, i);
        if pos.x >= 0.0 && u32(pos.x) < vars_data.width && pos.y >= 0 && u32(pos.y) < vars_data.height {
            let idx = u32(pos.y) * vars_data.width + u32(pos.x);
            atomicAdd(&counts[idx], 1u);
        }
    }
}

@compute
@workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
    //num_workgroups.x = 100,
    //num_workgroups.y = 50
    //prng_data length = 640000
    //global_id.x goes from 0 to 6400-1
    //global_id.y goes from 0 to 50-1
    //idx1 goes from 0 to 49*6400+6399 = 319999
    //idx2 = 100*50*64/2

    let idx1 = global_id.y * num_workgroups.x*64 + global_id.x;
    let idx2 = num_workgroups.x * num_workgroups.y*64 + idx1;
    let re = prng_data[idx1];
    let im = prng_data[idx2];
    buddhabrot_iterations(re, im);
}
