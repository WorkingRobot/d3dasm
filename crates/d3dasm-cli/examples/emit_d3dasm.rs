//! Emit lossless `.d3dasm` text for every shader program in the input files.
//!
//! Each `.bin` may be an archive of many DXBC containers; one `.d3dasm` file is
//! written per shader program, named `<stem>.<NN>.<profile>.d3dasm`. Every file
//! is verified on the way out (assemble -> encode must reproduce the original
//! chunk bytes).
//!
//! Usage: cargo run --release --example emit_d3dasm -- <out_dir> <files...>

use std::path::Path;

fn main() {
    let mut args = std::env::args().skip(1);
    let out_dir = args
        .next()
        .expect("usage: emit_d3dasm <out_dir> <files...>");
    std::fs::create_dir_all(&out_dir).expect("create out dir");

    let mut emitted = 0usize;
    let mut failed = 0usize;

    for arg in args {
        let data = match std::fs::read(&arg) {
            Ok(d) => d,
            Err(e) => {
                eprintln!("skip {arg}: {e}");
                continue;
            }
        };
        let stem = Path::new(&arg)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "shader".into());

        let shaders = d3dasm::parse(&data);
        let mut idx = 0usize;
        for shader in &shaders {
            let Some(prog) = shader.program() else {
                continue;
            };

            let text = dxbc::serialize(prog);
            let original = dxbc::shex::encode(prog);

            // Verify the emitted text re-assembles to the same bytes.
            let verified = dxbc::assemble(&text)
                .map(|p| dxbc::shex::encode(&p) == original)
                .unwrap_or(false);

            let profile = format!(
                "{}_{}_{}",
                prog.shader_type, prog.major_version, prog.minor_version
            );
            let name = format!("{stem}.{idx:02}.{profile}.d3dasm");
            let path = Path::new(&out_dir).join(&name);
            if let Err(e) = std::fs::write(&path, &text) {
                eprintln!("write {}: {e}", path.display());
                failed += 1;
            } else if !verified {
                eprintln!("VERIFY FAILED: {name}");
                failed += 1;
            } else {
                emitted += 1;
            }
            idx += 1;
        }
    }

    println!("{emitted} .d3dasm files written to {out_dir} ({failed} failures)");
    if failed > 0 {
        std::process::exit(1);
    }
}
