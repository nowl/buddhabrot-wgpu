use std::path::Path;

use anyhow::Error;
use itertools::{repeat_n, Itertools};

use crate::png;

pub fn gather_data<P, N>(bundle_files: Vec<P>) -> Result<(u32, u32, u32, Vec<N>), Error>
where
    P: AsRef<Path>,
    N: num::Unsigned + num::Zero + Clone + num::PrimInt,
{
    let mut data = None;
    let mut width = 0;
    let mut height = 0;
    let mut iterations = 0;
    for bpath in bundle_files.iter() {
        let (w, h, i, partial_data) = png::read_bundle_data(&bpath)?;
        width = w;
        height = h;
        iterations = i;
        if data == None {
            data = Some(repeat_n(N::zero(), partial_data.len()).collect_vec());
        }

        let Some(ref mut d) = data else {
            unreachable!();
        };

        assert_eq!(d.len(), partial_data.len());

        d.iter_mut().zip(partial_data.iter()).for_each(|(a, b)| {
            let tmp = N::from(*b).unwrap();
            *a = *a + tmp;
        });
    }

    Ok((width, height, iterations, data.unwrap()))
}
