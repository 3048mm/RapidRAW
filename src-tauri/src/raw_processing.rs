use crate::image_processing::apply_orientation;
use anyhow::{Result, anyhow};
use image::{DynamicImage, ImageBuffer, Rgba};
use rawler::{
    decoders::{Orientation, RawDecodeParams},
    imgop::develop::{DemosaicAlgorithm, Intermediate, ProcessingStep, RawDevelop},
    rawimage::{RawImage, RawPhotometricInterpretation},
    rawsource::RawSource,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

pub fn develop_raw_image(
    file_bytes: &[u8],
    fast_demosaic: bool,
    highlight_compression: f32,
    linear_mode: String,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<DynamicImage> {
    let (developed_image, orientation) = develop_internal(
        file_bytes,
        fast_demosaic,
        highlight_compression,
        linear_mode,
        cancel_token,
    )?;
    Ok(apply_orientation(developed_image, orientation))
}

fn is_linear_raw_format(raw_image: &RawImage) -> bool {
    matches!(
        raw_image.photometric,
        RawPhotometricInterpretation::LinearRaw
    )
}

#[inline]
fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(3.0)
    }
}

#[inline]
fn smootherstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[inline]
fn recover_clipped_pixel(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max_c = r.max(g).max(b);

    if max_c <= 0.50 {
        return (r, g, b);
    }

    let mut cur_r = r;
    let mut cur_g = g;
    let mut cur_b = b;

    let outer_blend = smootherstep(0.50, 1.5, max_c);

    let magenta = (cur_r.min(cur_b) - cur_g).max(0.0);
    if magenta > 0.0 {
        let target_g = cur_r.min(cur_b) * 0.80 + ((cur_r + cur_b) * 0.5) * 0.20;
        let correction = (target_g - cur_g).max(0.0);
        cur_g += correction * outer_blend;
    }

    let residual = (cur_r.min(cur_b) - cur_g).max(0.0);
    if residual > 0.0 {
        cur_g += residual * outer_blend;
    }

    let new_max = cur_r.max(cur_g).max(cur_b);
    let min_c = cur_r.min(cur_g).min(cur_b);

    let knee = smoothstep(0.50, 1.5, new_max);

    if knee > 0.0 {
        let neutrality = (min_c / new_max.max(1e-5)).clamp(0.0, 1.0);

        let core_burn = smoothstep(0.60, 3.0, new_max);

        let desat = (knee * (neutrality * 0.85 + core_burn * 0.15)).clamp(0.0, 1.0);
        let smooth_desat = desat * desat * (3.0 - 2.0 * desat);

        let neutral_value = min_c + (new_max - min_c) * 1.0;

        cur_r = cur_r * (1.0 - smooth_desat) + neutral_value * smooth_desat;
        cur_g = cur_g * (1.0 - smooth_desat) + neutral_value * smooth_desat;
        cur_b = cur_b * (1.0 - smooth_desat) + neutral_value * smooth_desat;
    }

    (cur_r, cur_g, cur_b)
}

fn develop_internal(
    file_bytes: &[u8],
    fast_demosaic: bool,
    _highlight_compression: f32,
    linear_mode: String,
    cancel_token: Option<(Arc<AtomicUsize>, usize)>,
) -> Result<(DynamicImage, Orientation)> {
    let check_cancel = || -> Result<()> {
        if let Some((tracker, generation)) = &cancel_token
            && tracker.load(Ordering::SeqCst) != *generation
        {
            return Err(anyhow!("Load cancelled"));
        }
        Ok(())
    };

    check_cancel()?;

    let source = RawSource::new_from_slice(file_bytes);
    let decoder = rawler::get_decoder(&source)?;

    check_cancel()?;
    let mut raw_image: RawImage = decoder.raw_image(&source, &RawDecodeParams::default(), false)?;

    let metadata = decoder.raw_metadata(&source, &RawDecodeParams::default())?;
    let orientation = metadata
        .exif
        .orientation
        .map(Orientation::from_u16)
        .unwrap_or(Orientation::Normal);

    let is_linear_format = is_linear_raw_format(&raw_image);

    let (apply_ungamma, apply_calibration) = match linear_mode.as_str() {
        "gamma" => (true, true),
        "skip_calib" => (false, false),
        "gamma_skip_calib" => (true, false),
        _ => (false, true),
    };

    let original_white_level = raw_image
        .whitelevel
        .0
        .first()
        .cloned()
        .unwrap_or(u16::MAX as u32) as f32;
    let original_black_level = raw_image
        .blacklevel
        .levels
        .first()
        .map(|r| r.as_f32())
        .unwrap_or(0.0);

    for level in raw_image.whitelevel.0.iter_mut() {
        *level = u32::MAX;
    }

    let mut developer = RawDevelop::default();

    if is_linear_format {
        developer.steps.retain(|&step| {
            step != ProcessingStep::SRgb
                && step != ProcessingStep::Demosaic
                && (apply_calibration || step != ProcessingStep::Calibrate)
        });
    } else if fast_demosaic {
        developer.demosaic_algorithm = DemosaicAlgorithm::Speed;
        developer.steps.retain(|&step| step != ProcessingStep::SRgb);
    } else {
        developer.steps.retain(|&step| step != ProcessingStep::SRgb);
    }

    raw_image.wb_coeffs =
        crate::multi_exposure::neutralize_wb_if_multiexposure(raw_image.wb_coeffs, file_bytes);

    check_cancel()?;
    let mut developed_intermediate = developer.develop_intermediate(&raw_image)?;

    drop(raw_image);

    let denominator = (original_white_level - original_black_level).max(1.0);
    let rescale_factor = (u32::MAX as f32 - original_black_level) / denominator;

    let safe_highlight_compression = 1000.0;

    let clamp_limit = if fast_demosaic {
        1.0
    } else {
        safe_highlight_compression
    };

    let (width, height) = {
        let dim = developed_intermediate.dim();
        (dim.w as u32, dim.h as u32)
    };

    check_cancel()?;

    match &mut developed_intermediate {
        Intermediate::Monochrome(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                let mut linear_val = *p * rescale_factor;
                if is_linear_format && apply_ungamma {
                    linear_val = srgb_to_linear(linear_val.max(0.0));
                }
                *p = linear_val.clamp(0.0, clamp_limit);
            });
        }
        Intermediate::ThreeColor(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                let mut r = (p[0] * rescale_factor).max(0.0);
                let mut g = (p[1] * rescale_factor).max(0.0);
                let mut b = (p[2] * rescale_factor).max(0.0);

                if is_linear_format && apply_ungamma {
                    r = srgb_to_linear(r.max(0.0));
                    g = srgb_to_linear(g.max(0.0));
                    b = srgb_to_linear(b.max(0.0));
                }

                let (rec_r, rec_g, rec_b) = recover_clipped_pixel(r, g, b);

                p[0] = rec_r.clamp(0.0, clamp_limit);
                p[1] = rec_g.clamp(0.0, clamp_limit);
                p[2] = rec_b.clamp(0.0, clamp_limit);
            });
        }
        Intermediate::FourColor(pixels) => {
            pixels.data.iter_mut().for_each(|p| {
                p.iter_mut().for_each(|c| {
                    let mut linear_val = *c * rescale_factor;
                    if is_linear_format && apply_ungamma {
                        linear_val = srgb_to_linear(linear_val.max(0.0));
                    }
                    *c = linear_val.clamp(0.0, clamp_limit);
                });
            });
        }
    }

    check_cancel()?;

    let dynamic_image = match developed_intermediate {
        Intermediate::ThreeColor(pixels) => {
            let buffer = ImageBuffer::<Rgba<f32>, _>::from_fn(width, height, |x, y| {
                let p = pixels.data[(y * width + x) as usize];
                Rgba([p[0], p[1], p[2], 1.0])
            });
            DynamicImage::ImageRgba32F(buffer)
        }
        Intermediate::Monochrome(pixels) => {
            let buffer = ImageBuffer::<Rgba<f32>, _>::from_fn(width, height, |x, y| {
                let p = pixels.data[(y * width + x) as usize];
                Rgba([p, p, p, 1.0])
            });
            DynamicImage::ImageRgba32F(buffer)
        }
        _ => {
            return Err(anyhow!("Unsupported intermediate format for conversion"));
        }
    };

    Ok((dynamic_image, orientation))
}

pub fn get_fast_demosaic_scale_factor(
    file_bytes: &[u8],
    decoded_width: u32,
    decoded_height: u32,
) -> f32 {
    let source = RawSource::new_from_slice(file_bytes);
    if let Ok(decoder) = rawler::get_decoder(&source)
        && let Ok(raw_img) = decoder.raw_image(&source, &RawDecodeParams::default(), true)
    {
        let max_orig = (raw_img.width as f32).max(raw_img.height as f32);
        let max_comp = (decoded_width as f32).max(decoded_height as f32);
        if max_orig > 0.0 {
            let ratio = max_comp / max_orig;
            if ratio > 0.1 && ratio < 0.35 {
                return 0.25;
            } else if (0.35..0.75).contains(&ratio) {
                return 0.5;
            }
        }
    }
    1.0
}

#[cfg(test)]
mod highlight_probe {
    use super::*;

    fn v161(r: f32, g: f32, b: f32, hc: f32) -> (f32, f32, f32) {
        let max_c = r.max(g).max(b);
        if max_c <= 1.0 {
            return (r, g, b);
        }
        let min_c = r.min(g).min(b);
        let f = (1.0 - (max_c - 1.0) / (hc - 1.0)).clamp(0.0, 1.0);
        let (cr, cg, cb) = (
            min_c + (r - min_c) * f,
            min_c + (g - min_c) * f,
            min_c + (b - min_c) * f,
        );
        let m = cr.max(cg).max(cb);
        if m > 1e-6 {
            (cr * max_c / m, cg * max_c / m, cb * max_c / m)
        } else {
            (max_c, max_c, max_c)
        }
    }

    fn hue(r: f32, g: f32, b: f32) -> f32 {
        let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
        let d = mx - mn;
        if d < 1e-6 {
            return -1.0;
        }
        let h = if mx == r {
            ((g - b) / d).rem_euclid(6.0)
        } else if mx == g {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        };
        h * 60.0
    }

    fn develop(bytes: &[u8], calibrate: bool) -> (Vec<[f32; 3]>, [f32; 4], f32, f32) {
        let source = RawSource::new_from_slice(bytes);
        let decoder = rawler::get_decoder(&source).unwrap();
        let mut raw = decoder
            .raw_image(&source, &RawDecodeParams::default(), false)
            .unwrap();
        let wl = raw.whitelevel.0[0] as f32;
        let bl = raw.blacklevel.levels[0].as_f32();
        for l in raw.whitelevel.0.iter_mut() {
            *l = u32::MAX;
        }
        let wb = raw.wb_coeffs;
        let mut dev = RawDevelop::default();
        dev.demosaic_algorithm = DemosaicAlgorithm::Speed;
        dev.steps.retain(|&s| {
            s != ProcessingStep::SRgb && (calibrate || s != ProcessingStep::Calibrate)
        });
        let inter = dev.develop_intermediate(&raw).unwrap();
        let k = (u32::MAX as f32 - bl) / (wl - bl).max(1.0);
        let px = match inter {
            Intermediate::ThreeColor(p) => p
                .data
                .iter()
                .map(|q| [q[0] * k, q[1] * k, q[2] * k])
                .collect(),
            _ => panic!("not 3ch"),
        };
        (px, wb, wl, bl)
    }

    #[test]
    #[ignore]
    fn probe_lantern() {
        let path = std::env::var("PROBE_NEF").unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let (cam, wb, wl, bl) = develop(&bytes, false);
        let (out, ..) = develop(&bytes, true);
        println!(
            "wb={wb:?} white={wl} black={bl} n={} / {}",
            cam.len(),
            out.len()
        );
        let mx: Vec<f32> = cam.iter().map(|p| p[0].max(p[1]).max(p[2])).collect();
        println!("cam max sample: p99.9={:?}", {
            let mut v = mx.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            [v[v.len() * 999 / 1000], v[v.len() - 1]]
        });
        let th = 0.98;
        let mut groups: std::collections::BTreeMap<String, Vec<usize>> = Default::default();
        for (i, p) in cam.iter().enumerate() {
            let key = format!(
                "{}{}{}",
                if p[0] >= th { 'R' } else { '-' },
                if p[1] >= th { 'G' } else { '-' },
                if p[2] >= th { 'B' } else { '-' }
            );
            if key != "---" || out[i].iter().cloned().fold(0.0, f32::max) > 0.7 {
                groups.entry(key).or_default().push(i);
            }
        }
        for (key, idx) in &groups {
            let mut idx = idx.clone();
            idx.sort_by(|&a, &b| {
                let ma = out[a].iter().cloned().fold(0.0, f32::max);
                let mb = out[b].iter().cloned().fold(0.0, f32::max);
                ma.partial_cmp(&mb).unwrap()
            });
            println!("\n== clipped {key}: {} px", idx.len());
            for q in [0.1, 0.5, 0.9, 0.99] {
                let i = idx[((idx.len() - 1) as f32 * q) as usize];
                let c = cam[i];
                let o = out[i];
                let m = recover_clipped_pixel(o[0], o[1], o[2]);
                let v = v161(o[0], o[1], o[2], 2.5);
                println!(
                    "  q{q:<4} cam=({:.2},{:.2},{:.2}) out=({:.2},{:.2},{:.2}) h={:.0} | main=({:.2},{:.2},{:.2}) h={:.0} | v161=({:.2},{:.2},{:.2}) h={:.0}",
                    c[0],
                    c[1],
                    c[2],
                    o[0],
                    o[1],
                    o[2],
                    hue(o[0], o[1], o[2]),
                    m.0,
                    m.1,
                    m.2,
                    hue(m.0, m.1, m.2),
                    v.0,
                    v.1,
                    v.2,
                    hue(v.0, v.1, v.2)
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn probe_matrix() {
        use rawler::imgop::{
            matrix::{multiply, normalize, pseudo_inverse},
            xyz::SRGB_TO_XYZ_D65,
        };
        let bytes = std::fs::read(std::env::var("PROBE_NEF").unwrap()).unwrap();
        let source = RawSource::new_from_slice(&bytes);
        let decoder = rawler::get_decoder(&source).unwrap();
        let raw = decoder
            .raw_image(&source, &RawDecodeParams::default(), false)
            .unwrap();
        println!(
            "camera={} {} wb={:?}",
            raw.camera.make, raw.camera.model, raw.wb_coeffs
        );
        for (ill, m) in raw.color_matrix.iter() {
            println!("color_matrix[{ill:?}] (xyz2cam) = {m:?}");
        }
        let m = raw
            .color_matrix
            .iter()
            .find(|(i, _)| format!("{i:?}") == "D65")
            .map(|(_, m)| m.clone())
            .unwrap();
        let xyz2cam = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
        let rgb2cam = normalize(multiply(&xyz2cam, &SRGB_TO_XYZ_D65));
        let cam2rgb = pseudo_inverse(rgb2cam);
        println!("rgb2cam (normalized) = {rgb2cam:?}");
        println!("cam2rgb = {cam2rgb:?}");
        let wb = raw.wb_coeffs;
        for c in [[1.06f32, 0.25, 0.03], [1.06, 1.06, 1.06]] {
            let (r, g, b) = (c[0] * wb[0], c[1] * wb[1], c[2] * wb[2]);
            let o: Vec<f32> = (0..3)
                .map(|i| cam2rgb[i][0] * r + cam2rgb[i][1] * g + cam2rgb[i][2] * b)
                .collect();
            println!("cam {c:?} -> wb ({r:.3},{g:.3},{b:.3}) -> rgb {o:?}");
        }
    }

    fn render(
        bytes: &[u8],
        wb_override: Option<[f32; 4]>,
        cat: Option<[[f32; 3]; 3]>,
        exposure: f32,
        out: &str,
    ) {
        let source = RawSource::new_from_slice(bytes);
        let decoder = rawler::get_decoder(&source).unwrap();
        let mut raw = decoder
            .raw_image(&source, &RawDecodeParams::default(), false)
            .unwrap();
        let orientation = decoder
            .raw_metadata(&source, &RawDecodeParams::default())
            .unwrap()
            .exif
            .orientation
            .map(Orientation::from_u16)
            .unwrap_or(Orientation::Normal);
        let wl = raw.whitelevel.0[0] as f32;
        let bl = raw.blacklevel.levels[0].as_f32();
        for l in raw.whitelevel.0.iter_mut() {
            *l = u32::MAX;
        }
        if let Some(wb) = wb_override {
            raw.wb_coeffs = wb;
        }
        let mut dev = RawDevelop::default();
        dev.demosaic_algorithm = DemosaicAlgorithm::Speed;
        dev.steps.retain(|&s| s != ProcessingStep::SRgb);
        let inter = dev.develop_intermediate(&raw).unwrap();
        let (w, h) = {
            let d = inter.dim();
            (d.w as u32, d.h as u32)
        };
        let k = (u32::MAX as f32 - bl) / (wl - bl).max(1.0) * exposure;
        let px = match inter {
            Intermediate::ThreeColor(p) => p.data,
            _ => panic!(),
        };
        let enc = |v: f32| {
            let v = v.clamp(0.0, 1.0);
            let g = if v <= 0.0031308 {
                v * 12.92
            } else {
                1.055 * v.powf(1.0 / 2.4) - 0.055
            };
            (g * 255.0 + 0.5) as u8
        };
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            let p = px[(y * w + x) as usize];
            let mut c = [p[0] * k, p[1] * k, p[2] * k];
            if let Some(m) = cat {
                c = [0, 1, 2].map(|i| m[i][0] * c[0] + m[i][1] * c[1] + m[i][2] * c[2]);
            }
            image::Rgb([enc(c[0]), enc(c[1]), enc(c[2])])
        });
        let img = apply_orientation(DynamicImage::ImageRgb8(img), orientation);
        img.save(out).unwrap();
        println!("saved {out}");
    }

    fn mat_mul(a: [[f32; 3]; 3], b: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
        let mut r = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    r[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        r
    }

    fn mat_inv(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
        let d = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let c =
            |a: usize, b: usize, c2: usize, d2: usize| m[a][b] * m[c2][d2] - m[a][d2] * m[c2][b];
        [
            [c(1, 1, 2, 2) / d, -c(0, 1, 2, 2) / d, c(0, 1, 1, 2) / d],
            [-c(1, 0, 2, 2) / d, c(0, 0, 2, 2) / d, -c(0, 0, 1, 2) / d],
            [c(1, 0, 2, 1) / d, -c(0, 0, 2, 1) / d, c(0, 0, 1, 1) / d],
        ]
    }

    #[test]
    #[ignore]
    fn probe_render_d65() {
        let path = std::env::var("PROBE_NEF").unwrap();
        let exposure: f32 = std::env::var("PROBE_EV")
            .map(|v| 2f32.powf(v.parse().unwrap()))
            .unwrap_or(1.0);
        let bytes = std::fs::read(&path).unwrap();
        let source = RawSource::new_from_slice(&bytes);
        let decoder = rawler::get_decoder(&source).unwrap();
        let raw = decoder
            .raw_image(&source, &RawDecodeParams::default(), false)
            .unwrap();
        let m = raw
            .color_matrix
            .iter()
            .find(|(i, _)| format!("{i:?}") == "D65")
            .map(|(_, m)| m.clone())
            .unwrap();
        let xyz2cam = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
        let d65 = [0.95047f32, 1.0, 1.08883];
        let cam_d65 = [0, 1, 2]
            .map(|i| xyz2cam[i][0] * d65[0] + xyz2cam[i][1] * d65[1] + xyz2cam[i][2] * d65[2]);
        let wb_d65 = [
            cam_d65[1] / cam_d65[0],
            1.0,
            cam_d65[1] / cam_d65[2],
            f32::NAN,
        ];
        let shot = raw.wb_coeffs;
        let cam2xyz = mat_inv(xyz2cam);
        let n = [1.0 / shot[0], 1.0 / shot[1], 1.0 / shot[2]];
        let src_xyz =
            [0, 1, 2].map(|i| cam2xyz[i][0] * n[0] + cam2xyz[i][1] * n[1] + cam2xyz[i][2] * n[2]);
        let src_xyz = src_xyz.map(|v| v / src_xyz[1]);
        let sum: f32 = src_xyz.iter().sum();
        println!(
            "wb_shot={shot:?} wb_d65={wb_d65:?} source white xy=({:.4},{:.4})",
            src_xyz[0] / sum,
            src_xyz[1] / sum
        );
        let m16 = [
            [0.401288, 0.650173, -0.051461],
            [-0.250268, 1.204414, 0.045854],
            [-0.002079, 0.048952, 0.953127],
        ];
        let rgb2xyz = [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ];
        let ls = [0, 1, 2]
            .map(|i| m16[i][0] * src_xyz[0] + m16[i][1] * src_xyz[1] + m16[i][2] * src_xyz[2]);
        let ld = [0, 1, 2].map(|i| m16[i][0] * d65[0] + m16[i][1] * d65[1] + m16[i][2] * d65[2]);
        let diag = [
            [ld[0] / ls[0], 0.0, 0.0],
            [0.0, ld[1] / ls[1], 0.0],
            [0.0, 0.0, ld[2] / ls[2]],
        ];
        let cat = mat_mul(
            mat_inv(rgb2xyz),
            mat_mul(mat_inv(m16), mat_mul(diag, mat_mul(m16, rgb2xyz))),
        );
        println!("cat (linear sRGB) = {cat:?}");
        let dir = std::env::var("PROBE_OUT").unwrap();
        render(
            &bytes,
            None,
            None,
            exposure,
            &format!("{dir}/probe_A_asshot.png"),
        );
        render(
            &bytes,
            Some(wb_d65),
            None,
            exposure,
            &format!("{dir}/probe_B_d65_nocat.png"),
        );
        render(
            &bytes,
            Some(wb_d65),
            Some(cat),
            exposure,
            &format!("{dir}/probe_C_d65_cat16.png"),
        );
    }

    fn save_srgb(img: &DynamicImage, out: &str) {
        let f = img.to_rgb32f();
        let enc = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
        let o = image::RgbImage::from_fn(f.width(), f.height(), |x, y| {
            let p = f.get_pixel(x, y);
            image::Rgb([enc(p[0]), enc(p[1]), enc(p[2])])
        });
        o.save(out).unwrap();
        println!("saved {out}");
    }

    #[test]
    #[ignore]
    fn probe_render_agx() {
        let bytes = std::fs::read(std::env::var("PROBE_NEF").unwrap()).unwrap();
        let dir = std::env::var("PROBE_OUT").unwrap();
        let mut img = develop_raw_image(&bytes, false, 2.5, "linear".to_string(), None).unwrap();
        let (w, h) = (img.width(), img.height());
        img = img.resize(w / 3, h / 3, image::imageops::FilterType::Triangle);
        crate::image_processing::apply_cpu_agx_tonemap(&mut img);
        save_srgb(&img, &format!("{dir}/probe_D_main_decode_agx.png"));

        let (px, ..) = develop(&bytes, true);
        let source = RawSource::new_from_slice(&bytes);
        let decoder = rawler::get_decoder(&source).unwrap();
        let orientation = decoder
            .raw_metadata(&source, &RawDecodeParams::default())
            .unwrap()
            .exif
            .orientation
            .map(Orientation::from_u16)
            .unwrap_or(Orientation::Normal);
        let raw = decoder
            .raw_image(&source, &RawDecodeParams::default(), false)
            .unwrap();
        let mut dev = RawDevelop::default();
        dev.demosaic_algorithm = DemosaicAlgorithm::Speed;
        let d = dev.develop_intermediate(&raw).unwrap().dim();
        let (w2, h2) = (d.w as u32, d.h as u32);
        let buf = ImageBuffer::<Rgba<f32>, _>::from_fn(w2, h2, |x, y| {
            let p = px[(y * w2 + x) as usize];
            Rgba([p[0].max(0.0), p[1].max(0.0), p[2].max(0.0), 1.0])
        });
        let mut e = apply_orientation(DynamicImage::ImageRgba32F(buf), orientation);
        crate::image_processing::apply_cpu_agx_tonemap(&mut e);
        save_srgb(&e, &format!("{dir}/probe_E_norecover_agx.png"));
        let mut r = e.clone();
        let buf = ImageBuffer::<Rgba<f32>, _>::from_fn(w2, h2, |x, y| {
            let p = px[(y * w2 + x) as usize];
            let (a, b, c) = recover_clipped_pixel(p[0].max(0.0), p[1].max(0.0), p[2].max(0.0));
            Rgba([a, b, c, 1.0])
        });
        r = apply_orientation(DynamicImage::ImageRgba32F(buf), orientation);
        crate::image_processing::apply_cpu_agx_tonemap(&mut r);
        save_srgb(&r, &format!("{dir}/probe_F_recover_agx_speed.png"));
    }

    fn reconstruct(cam: &mut [[f32; 3]], wb: [f32; 4], clip: f32) {
        let clip_wb = [clip * wb[0], clip * wb[1], clip * wb[2]];
        for p in cam.iter_mut() {
            for c in 0..3 {
                p[c] *= wb[c];
            }
        }
        for c in 0..3 {
            let (o1, o2) = ((c + 1) % 3, (c + 2) % 3);
            let mut ratios: Vec<f32> = cam
                .iter()
                .filter_map(|p| {
                    let v = p[c];
                    let opp = 0.5 * (p[o1] + p[o2]);
                    (v >= 0.85 * clip_wb[c]
                        && v < 0.98 * clip_wb[c]
                        && p[o1] < 0.98 * clip_wb[o1]
                        && p[o2] < 0.98 * clip_wb[o2]
                        && opp > 0.01 * clip_wb[c])
                        .then_some(v / opp)
                })
                .collect();
            let n_clip = cam.iter().filter(|p| p[c] >= 0.98 * clip_wb[c]).count();
            if ratios.len() < 100 || n_clip == 0 {
                println!("ch{c}: clipped={n_clip} samples={} -> skip", ratios.len());
                continue;
            }
            ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let ratio = ratios[ratios.len() / 2];
            println!(
                "ch{c}: clipped={n_clip} samples={} ratio(median)={ratio:.3} p10={:.3} p90={:.3}",
                ratios.len(),
                ratios[ratios.len() / 10],
                ratios[ratios.len() * 9 / 10]
            );
            let mut maxv = 0.0f32;
            for p in cam.iter_mut() {
                if p[c] >= 0.98 * clip_wb[c]
                    && p[o1] < 0.98 * clip_wb[o1]
                    && p[o2] < 0.98 * clip_wb[o2]
                {
                    let est = 0.5 * (p[o1] + p[o2]) * ratio;
                    let w = ((p[c] / clip_wb[c] - 0.98) / 0.02).clamp(0.0, 1.0);
                    p[c] = p[c].max(p[c] + (est - p[c]).max(0.0) * w);
                    maxv = maxv.max(p[c]);
                }
            }
            println!(
                "ch{c}: reconstructed max={maxv:.3} (clip level {:.3})",
                clip_wb[c]
            );
        }
    }

    fn agx_from_cam(
        cam: &[[f32; 3]],
        w: u32,
        h: u32,
        cam2rgb: [[f32; 3]; 3],
        orientation: Orientation,
        out: &str,
    ) {
        let buf = ImageBuffer::<Rgba<f32>, _>::from_fn(w, h, |x, y| {
            let p = cam[(y * w + x) as usize];
            let o = [0, 1, 2]
                .map(|i| cam2rgb[i][0] * p[0] + cam2rgb[i][1] * p[1] + cam2rgb[i][2] * p[2]);
            Rgba([o[0], o[1], o[2], 1.0])
        });
        let mut img = apply_orientation(DynamicImage::ImageRgba32F(buf), orientation);
        crate::image_processing::apply_cpu_agx_tonemap(&mut img);
        save_srgb(&img, out);
    }

    #[test]
    #[ignore]
    fn probe_reconstruct() {
        use rawler::imgop::{
            matrix::{multiply, normalize, pseudo_inverse},
            xyz::SRGB_TO_XYZ_D65,
        };
        let bytes = std::fs::read(std::env::var("PROBE_NEF").unwrap()).unwrap();
        let dir = std::env::var("PROBE_OUT").unwrap();
        let source = RawSource::new_from_slice(&bytes);
        let decoder = rawler::get_decoder(&source).unwrap();
        let orientation = decoder
            .raw_metadata(&source, &RawDecodeParams::default())
            .unwrap()
            .exif
            .orientation
            .map(Orientation::from_u16)
            .unwrap_or(Orientation::Normal);
        let raw = decoder
            .raw_image(&source, &RawDecodeParams::default(), false)
            .unwrap();
        let m = raw
            .color_matrix
            .iter()
            .find(|(i, _)| format!("{i:?}") == "D65")
            .map(|(_, m)| m.clone())
            .unwrap();
        let xyz2cam = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
        let cam2rgb = pseudo_inverse(normalize(multiply(&xyz2cam, &SRGB_TO_XYZ_D65)));
        let mut dev = RawDevelop::default();
        dev.demosaic_algorithm = DemosaicAlgorithm::Speed;
        let d = dev.develop_intermediate(&raw).unwrap().dim();
        let (w, h) = (d.w as u32, d.h as u32);
        let (cam, wb, ..) = develop(&bytes, false);
        let clip = cam
            .iter()
            .map(|p| p[0].max(p[1]).max(p[2]))
            .fold(0.0f32, f32::max);
        println!("clip level (camera RGB) = {clip:.4}");
        let mut plain = cam.clone();
        for p in plain.iter_mut() {
            for c in 0..3 {
                p[c] *= wb[c];
            }
        }
        let t = std::time::Instant::now();
        let mut rec = cam.clone();
        reconstruct(&mut rec, wb, clip);
        println!("reconstruct time: {:?} for {} px", t.elapsed(), rec.len());
        agx_from_cam(
            &plain,
            w,
            h,
            cam2rgb,
            orientation,
            &format!("{dir}/probe_I_plain_agx.png"),
        );
        agx_from_cam(
            &rec,
            w,
            h,
            cam2rgb,
            orientation,
            &format!("{dir}/probe_J_reconstructed_agx.png"),
        );
    }

    #[test]
    #[ignore]
    fn probe_rim() {
        use rawler::imgop::{
            matrix::{multiply, normalize, pseudo_inverse},
            xyz::SRGB_TO_XYZ_D65,
        };
        let bytes = std::fs::read(std::env::var("PROBE_NEF").unwrap()).unwrap();
        let dir = std::env::var("PROBE_OUT").unwrap();
        let source = RawSource::new_from_slice(&bytes);
        let decoder = rawler::get_decoder(&source).unwrap();
        let orientation = decoder
            .raw_metadata(&source, &RawDecodeParams::default())
            .unwrap()
            .exif
            .orientation
            .map(Orientation::from_u16)
            .unwrap_or(Orientation::Normal);
        let raw = decoder
            .raw_image(&source, &RawDecodeParams::default(), false)
            .unwrap();
        let m = raw
            .color_matrix
            .iter()
            .find(|(i, _)| format!("{i:?}") == "D65")
            .map(|(_, m)| m.clone())
            .unwrap();
        let xyz2cam = [[m[0], m[1], m[2]], [m[3], m[4], m[5]], [m[6], m[7], m[8]]];
        let cam2rgb = pseudo_inverse(normalize(multiply(&xyz2cam, &SRGB_TO_XYZ_D65)));
        let mut dev = RawDevelop::default();
        dev.demosaic_algorithm = DemosaicAlgorithm::Speed;
        let d = dev.develop_intermediate(&raw).unwrap().dim();
        let (w, h) = (d.w as u32, d.h as u32);
        let (cam, wb, ..) = develop(&bytes, false);
        let to_rgb = |p: [f32; 3]| {
            let q = [p[0] * wb[0], p[1] * wb[1], p[2] * wb[2]];
            [0, 1, 2].map(|i| cam2rgb[i][0] * q[0] + cam2rgb[i][1] * q[1] + cam2rgb[i][2] * q[2])
        };
        let variants: [(&str, fn([f32; 3]) -> [f32; 3]); 3] = [
            ("K_clipneg", |o| o.map(|v| v.max(0.0))),
            ("L_clipneg_recover", |o| {
                let o = o.map(|v| v.max(0.0));
                let r = recover_clipped_pixel(o[0], o[1], o[2]);
                [r.0, r.1, r.2]
            }),
            ("M_keepneg_recover", |o| {
                let r = recover_clipped_pixel(o[0], o[1], o[2]);
                [r.0, r.1, r.2]
            }),
        ];
        let mut neg = 0usize;
        for p in &cam {
            let o = to_rgb(*p);
            if o.iter().any(|v| *v < 0.0) {
                neg += 1;
            }
        }
        println!(
            "pixels with negative channel after matrix: {neg} / {} ({:.1}%)",
            cam.len(),
            neg as f32 * 100.0 / cam.len() as f32
        );
        for (name, f) in variants {
            let buf = ImageBuffer::<Rgba<f32>, _>::from_fn(w, h, |x, y| {
                let o = f(to_rgb(cam[(y * w + x) as usize]));
                Rgba([o[0], o[1], o[2], 1.0])
            });
            let mut img = apply_orientation(DynamicImage::ImageRgba32F(buf), orientation);
            crate::image_processing::apply_cpu_agx_tonemap(&mut img);
            save_srgb(&img, &format!("{dir}/probe_{name}_agx.png"));
        }
        for c in [[0.5f32, 0.03, 0.005], [0.8, 0.06, 0.01], [1.0, 0.1, 0.015]] {
            let o = to_rgb(c);
            println!("rim cam {c:?} -> rgb ({:.3},{:.3},{:.3})", o[0], o[1], o[2]);
        }
    }

    #[test]
    #[ignore]
    fn probe_preprocess() {
        let bytes = std::fs::read(std::env::var("PROBE_NEF").unwrap()).unwrap();
        let dir = std::env::var("PROBE_OUT").unwrap();
        let base = develop_raw_image(&bytes, false, 2.5, "linear".to_string(), None).unwrap();
        for (name, nr, sh) in [
            ("N_nr_sharpen", 14.0f32, 0.35f32),
            ("O_nr_only", 14.0, 0.0),
            ("P_sharpen_only", 0.0, 0.35),
            ("Q_none", 0.0, 0.0),
        ] {
            let mut img = base.clone();
            if nr > 0.0 || sh > 0.0 {
                crate::image_processing::remove_raw_artifacts_and_enhance(&mut img, nr, sh);
            }
            let (w, h) = (img.width(), img.height());
            let mut img = img.resize(w / 3, h / 3, image::imageops::FilterType::Triangle);
            crate::image_processing::apply_cpu_agx_tonemap(&mut img);
            save_srgb(&img, &format!("{dir}/probe_{name}_agx.png"));
            let crop = img.crop_imm(w / 3 * 2 / 5, 0, w / 3 * 3 / 10, h / 3 / 6);
            save_srgb(&crop, &format!("{dir}/probe_{name}_crop.png"));
        }
    }

    fn basic_tonemap(img: &DynamicImage) -> DynamicImage {
        let f = img.to_rgb32f();
        let enc = |v: f32| {
            let c = v.clamp(0.0, 1.0);
            let s = if c <= 0.0031308 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            };
            let s = s.powf(1.0 / 1.1);
            let cc = s * s * (3.0 - 2.0 * s);
            s + (cc - s) * 0.75
        };
        let o = image::ImageBuffer::<image::Rgb<f32>, _>::from_fn(f.width(), f.height(), |x, y| {
            let p = f.get_pixel(x, y);
            image::Rgb([enc(p[0]), enc(p[1]), enc(p[2])])
        });
        DynamicImage::ImageRgb32F(o)
    }

    #[test]
    #[ignore]
    fn probe_basic() {
        let bytes = std::fs::read(std::env::var("PROBE_NEF").unwrap()).unwrap();
        let dir = std::env::var("PROBE_OUT").unwrap();
        let mut img = develop_raw_image(&bytes, false, 2.5, "linear".to_string(), None).unwrap();
        crate::image_processing::remove_raw_artifacts_and_enhance(&mut img, 14.0, 0.35);
        let (w, h) = (img.width(), img.height());
        let img = img.resize(w / 3, h / 3, image::imageops::FilterType::Triangle);
        save_srgb(&basic_tonemap(&img), &format!("{dir}/probe_R_basic.png"));
    }

    #[test]
    #[ignore]
    fn probe_tonemap_table() {
        let samples: [(&str, [f32; 3]); 8] = [
            ("grey 0.18", [0.18, 0.18, 0.18]),
            ("grey 1.0", [1.0, 1.0, 1.0]),
            ("grey 4.0", [4.0, 4.0, 4.0]),
            ("rim dark", [0.5, 0.02, 0.01]),
            ("rim", [1.4, 0.0, 0.01]),
            ("rim bright", [2.2, 0.0, 0.02]),
            ("core (R clipped)", [2.8, 0.18, 0.16]),
            ("core reconstructed", [4.9, 0.3, 0.2]),
        ];
        let basic = |v: f32| {
            let c = v.clamp(0.0, 1.0);
            let s = if c <= 0.0031308 {
                c * 12.92
            } else {
                1.055 * c.powf(1.0 / 2.4) - 0.055
            };
            let s = s.powf(1.0 / 1.1);
            let cc = s * s * (3.0 - 2.0 * s);
            s + (cc - s) * 0.75
        };
        for (name, c) in samples {
            let buf = ImageBuffer::<Rgba<f32>, _>::from_pixel(1, 1, Rgba([c[0], c[1], c[2], 1.0]));
            let mut img = DynamicImage::ImageRgba32F(buf);
            crate::image_processing::apply_cpu_agx_tonemap(&mut img);
            let a = img.to_rgb32f();
            let a = a.get_pixel(0, 0);
            let to8 = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
            println!(
                "ROW|{name}|{:?}|{},{},{}|{},{},{}",
                c,
                to8(basic(c[0])),
                to8(basic(c[1])),
                to8(basic(c[2])),
                to8(a[0]),
                to8(a[1]),
                to8(a[2])
            );
        }
        let mut curve = String::new();
        for i in 0..=48 {
            let ev = -8.0 + i as f32 * 0.25;
            let v = 0.18 * 2f32.powf(ev);
            let buf = ImageBuffer::<Rgba<f32>, _>::from_pixel(1, 1, Rgba([v, v, v, 1.0]));
            let mut img = DynamicImage::ImageRgba32F(buf);
            crate::image_processing::apply_cpu_agx_tonemap(&mut img);
            let a = img.to_rgb32f().get_pixel(0, 0)[0];
            curve.push_str(&format!("[{ev},{:.4},{:.4}],", basic(v), a));
        }
        println!("CURVE|{curve}");
    }
}
