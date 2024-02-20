use anyhow::Error;
use std::{
    fs::File,
    io::{BufWriter, Read},
    path::Path,
    time::SystemTime,
};
use zip::ZipArchive;

pub fn read_bundle_data<P>(bpath: P) -> Result<(u32, u32, u32, Vec<u32>), Error>
where
    P: AsRef<Path>,
{
    let bfile = File::open(bpath)?;
    let mut zip = ZipArchive::new(&bfile)?;

    let mut datafile = zip.by_name("data.bin")?;
    println!("Filename: {}", datafile.name());

    let mut buf = [0; 1];
    datafile.read_exact(&mut buf)?;
    assert!(buf[0] == 0x01);

    let mut buf = [0; 4];
    datafile.read_exact(&mut buf)?;
    let iterations = u32::from_le_bytes(buf);

    datafile.read_exact(&mut buf)?;
    let width = u32::from_le_bytes(buf);

    datafile.read_exact(&mut buf)?;
    let height = u32::from_le_bytes(buf);

    let mut buf = Vec::with_capacity(width as usize * height as usize * std::mem::size_of::<u32>());
    datafile.read_to_end(&mut buf)?;

    let data = bytemuck::cast_slice::<u8, u32>(&buf)
        .into_iter()
        .cloned()
        .collect::<Vec<u32>>();

    assert!(data.len() as u32 == width * height);

    Ok((width, height, iterations, data))
}

pub fn write_png(width: u32, height: u32, input_data: &Vec<u64>) -> Result<(), Error> {
    let since_epoch = SystemTime::now().duration_since(SystemTime::UNIX_EPOCH)?;
    let filename = format!("{}.png", since_epoch.as_millis(),);

    let path = Path::new(&filename);
    let file = File::create(path).unwrap();
    let ref mut w = BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Sixteen);
    let mut writer = encoder.write_header().unwrap();

    let max = input_data.iter().max().unwrap().clone();
    let mut data = Vec::with_capacity(input_data.len() * 3 * std::mem::size_of::<u16>());
    for d in input_data.iter() {
        let v = d * 0xffff / max;
        assert!(v <= 0xffff);
        let bytes = (v as u16).to_be_bytes();
        data.extend_from_slice(&bytes);
        data.extend_from_slice(&bytes);
        data.extend_from_slice(&bytes);
    }

    writer.write_image_data(&data).unwrap();

    Ok(())
}
