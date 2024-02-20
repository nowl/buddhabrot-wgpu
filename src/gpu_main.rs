use anyhow::Error;

mod fractal;
mod gpu;

use clap::Parser;
use num::Complex;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Name of prefix on bbundle output file
    #[arg(short, long)]
    name: String,

    /// Width of output in pixels
    #[arg(long, default_value_t = 320)]
    width: u32,

    /// Height of output in pixels
    #[arg(long, default_value_t = 240)]
    height: u32,

    /// Max iterations
    #[arg(short, long, default_value_t = 1000)]
    iterations: u32,

    /// Number of parallel trials to run on the GPU each iteration
    #[arg(long, default_value_t = 6400*10)]
    gpu_trials: u32,

    /// Number of times to run per zip file
    #[arg(short, long, default_value_t = 10)]
    runs_per_zip: u32,

    /// Real part of full image lower left corner
    #[arg(long, default_value_t = -2.25, allow_hyphen_values = true)]
    lower_left_re: f32,

    /// Imaginary part of full image lower left corner
    #[arg(long, default_value_t = -1.5, allow_hyphen_values = true)]
    lower_left_im: f32,

    /// Real part of full image upper right corner
    #[arg(long, default_value_t = 1.0, allow_hyphen_values = true)]
    upper_right_re: f32,

    /// Imaginary part of full image upper right corner
    #[arg(long, default_value_t = 1.5, allow_hyphen_values = true)]
    upper_right_im: f32,

    /// Real part of zoom lower left corner
    #[arg(long, default_value_t = -2.25, allow_hyphen_values = true)]
    zoom_lower_left_re: f32,

    /// Imaginary part of zoom lower left corner
    #[arg(long, default_value_t = -1.5, allow_hyphen_values = true)]
    zoom_lower_left_im: f32,

    /// Real part of zoom upper right corner
    #[arg(long, default_value_t = 1.0, allow_hyphen_values = true)]
    zoom_upper_right_re: f32,

    /// Imaginary part of zoom upper right corner
    #[arg(long, default_value_t = 1.5, allow_hyphen_values = true)]
    zoom_upper_right_im: f32,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();

    env_logger::init();

    let ll = Complex::new(args.lower_left_re, args.lower_left_im);
    let ur = Complex::new(args.upper_right_re, args.upper_right_im);
    let llz = Complex::new(args.zoom_lower_left_re, args.zoom_lower_left_im);
    let urz = Complex::new(args.zoom_upper_right_re, args.zoom_upper_right_im);
    let mut buddhabrot_gpu = fractal::BuddhabrotGPU::new(
        args.width,
        args.height,
        args.iterations,
        args.gpu_trials,
        ll,
        ur,
        llz,
        urz,
    );

    let mut run_count = 1;

    loop {
        for trial in 0..args.runs_per_zip {
            buddhabrot_gpu.update();
            //buddhabrot_gpu.dump_stats();
            println!(
                "Trial: {}/{}, Run: {}",
                trial + 1,
                args.runs_per_zip,
                run_count
            );
        }
        log::info!("Writing zip number {}", run_count);
        buddhabrot_gpu.dump_to_file(&args.name)?;
        buddhabrot_gpu.reset();

        run_count += 1;
    }

    Ok(())
}
