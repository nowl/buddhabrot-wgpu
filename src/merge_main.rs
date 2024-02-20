use anyhow::Error;

mod bundle;
mod fractal;
mod gpu;

use clap::Parser;

use glob::glob;

mod png;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// File glob of bbundle files to include in output
    #[arg(short, long)]
    bundle_files: String,

    /// Name of prefix on bbundle output file
    #[arg(short, long)]
    name: String,
}

fn main() -> Result<(), Error> {
    let args = Args::parse();

    env_logger::init();

    let bundle_files = glob(&args.bundle_files)
        .expect("Failed to read glob pattern")
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;

    log::info!("bundle files: {:?}", bundle_files);

    let (width, height, iterations, data) = bundle::gather_data(bundle_files)?;

    fractal::dump_to_file(width, height, iterations, &data, &args.name)?;

    Ok(())
}
