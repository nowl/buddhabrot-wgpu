use anyhow::Error;
use std::{fs::File, path::Path, sync::Mutex, time::SystemTime};

use itertools::{izip, Itertools};
use num::complex::Complex;
use rand::{thread_rng, Rng};

use crate::gpu::GPUHandle;

pub struct BuddhabrotGPU {
    gpu: GPUHandle,
    num_trials_x2: u32,
    width: u32,
    height: u32,
    lower_left: Complex<f32>,
    upper_right: Complex<f32>,
    zoom_lower_left: Complex<f32>,
    zoom_upper_right: Complex<f32>,
    pub frame: Vec<u32>,
    max_iters: u32,
}

fn get_rng_block(n: u32) -> Vec<f32> {
    let mut r = thread_rng();
    let mut v = vec![];
    for _ in 0..n {
        v.push(r.gen());
    }
    v
}

impl BuddhabrotGPU {
    pub fn new(
        width: u32,
        height: u32,
        max_iters: u32,
        gpu_trials: u32,
        lower_left: Complex<f32>,
        upper_right: Complex<f32>,
        zoom_lower_left: Complex<f32>,
        zoom_upper_right: Complex<f32>,
    ) -> Self {
        let num_trials_x2 = (gpu_trials * 2).next_multiple_of(6400);
        let gpu = GPUHandle::new(num_trials_x2, width, height);

        Self {
            gpu,
            num_trials_x2,
            width,
            height,
            lower_left,
            upper_right,
            zoom_lower_left,
            zoom_upper_right,
            frame: vec![0; (width * height) as usize],
            max_iters,
        }
    }

    pub fn reset(&mut self) {
        self.frame.iter_mut().for_each(|x| *x = 0);
    }

    pub fn dump_stats(&self) {
        let sum = self.frame.iter().map(|x| *x as u64).sum::<u64>();
        let max = self.frame.iter().map(|x| *x as u64).max();

        log::info!("sum: {sum}, minmax: {max:?}");
    }

    pub fn update(&mut self) {
        let result = self.gpu.call(
            self.lower_left,
            self.upper_right,
            self.zoom_lower_left,
            self.zoom_upper_right,
            self.max_iters,
            get_rng_block(self.num_trials_x2),
        );
        for (n, v) in self.frame.iter_mut().enumerate() {
            let c = result[n];
            *v += c;
        }
    }

    pub fn dump_to_file(&self, prefix: &str) -> Result<(), Error> {
        dump_to_file(self.width, self.height, self.max_iters, &self.frame, prefix)
    }
}

pub fn dump_to_file(
    width: u32,
    height: u32,
    max_iters: u32,
    frame: &Vec<u32>,
    prefix: &str,
) -> Result<(), Error> {
    use std::io::Write;

    let since_epoch = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?;
    let filename = format!(
        "bbundle_{}_{}_{}_{}_{}.zip",
        prefix,
        since_epoch.as_millis(),
        width,
        height,
        max_iters
    );
    log::info!("name: {}", filename);

    let mut zip = zip::ZipWriter::new(File::create(Path::new(&filename))?);

    let options = zip::write::FileOptions::default()
        .large_file(true)
        .compression_method(zip::CompressionMethod::Deflated);
    let filename = format!("data.bin");
    zip.start_file(filename, options)?;

    zip.write_all(&[0x01])?;
    zip.write_all(&max_iters.to_le_bytes())?;
    zip.write_all(&width.to_le_bytes())?;
    zip.write_all(&height.to_le_bytes())?;

    let data_u8 = bytemuck::cast_slice(frame);
    zip.write_all(data_u8)?;

    zip.finish()?;

    Ok(())
}
