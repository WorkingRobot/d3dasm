use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

use clap::Parser;

/// Direct3D shader bytecode disassembler and `.d3dasm` assembler.
///
/// Default mode disassembles a DXBC binary. With `--emit d3dasm` it writes the
/// lossless `.d3dasm` text format; with `--assemble` it reads a `.d3dasm` file
/// and re-encodes it to a SHEX shader chunk (byte-identical to the original).
#[derive(Parser)]
#[command(name = "d3dasm", version, about)]
struct Cli {
    /// Input file: a DXBC binary, or a `.d3dasm` text file with `--assemble`.
    file: PathBuf,

    /// Assemble a `.d3dasm` file back into raw SHEX shader-chunk bytecode.
    #[arg(long)]
    assemble: bool,

    /// Output format when disassembling: `human` (default) or `d3dasm` (lossless).
    #[arg(long, value_name = "FORMAT", default_value = "human")]
    emit: String,

    /// Write output to a file instead of stdout.
    #[arg(short, long)]
    output: Option<PathBuf>,
}

fn main() {
    let cli = Cli::parse();

    let result = if cli.assemble {
        assemble(&cli)
    } else {
        disassemble(&cli)
    };

    if let Err(msg) = result {
        eprintln!("{msg}");
        process::exit(1);
    }
}

/// Disassemble a DXBC binary to human text or lossless `.d3dasm`.
fn disassemble(cli: &Cli) -> Result<(), String> {
    let data = std::fs::read(&cli.file)
        .map_err(|e| format!("Error reading {}: {e}", cli.file.display()))?;
    let shaders = d3dasm::parse(&data);
    if shaders.is_empty() {
        return Err(format!(
            "No DXBC shader bytecode found in {}",
            cli.file.display()
        ));
    }

    let mut out = String::new();
    match cli.emit.as_str() {
        "human" => {
            out.push_str(&format!("// File: {}\n", cli.file.display()));
            out.push_str(&format!("// Found {} DXBC shader(s)\n\n", shaders.len()));
            for (i, shader) in shaders.iter().enumerate() {
                out.push_str("// ============================================================\n");
                out.push_str(&format!(
                    "// Shader #{i}: DXBC at 0x{:X}, size={}\n",
                    shader.offset(),
                    shader.size()
                ));
                out.push_str("// ============================================================\n");
                out.push_str(&format!("{shader}\n"));
            }
        }
        "d3dasm" => {
            let mut programs = shaders.iter().filter_map(|s| s.program());
            let program = programs
                .next()
                .ok_or("input has no shader program (SHEX/SHDR chunk) to serialize")?;
            if programs.next().is_some() {
                return Err(
                    "input has multiple shader programs; .d3dasm represents a single program"
                        .into(),
                );
            }
            out.push_str(&d3dasm::dxbc::serialize(program));
        }
        other => {
            return Err(format!(
                "unknown --emit format: {other:?} (use human|d3dasm)"
            ));
        }
    }

    write_out(cli.output.as_deref(), out.as_bytes())
}

/// Assemble a `.d3dasm` text file into raw SHEX shader-chunk bytecode.
fn assemble(cli: &Cli) -> Result<(), String> {
    let text = std::fs::read_to_string(&cli.file)
        .map_err(|e| format!("Error reading {}: {e}", cli.file.display()))?;
    let program = d3dasm::dxbc::assemble(&text).map_err(|e| e.to_string())?;
    let bytes = d3dasm::dxbc::shex::encode(&program);
    write_out(cli.output.as_deref(), &bytes)
}

fn write_out(output: Option<&Path>, bytes: &[u8]) -> Result<(), String> {
    match output {
        Some(path) => std::fs::write(path, bytes)
            .map_err(|e| format!("Error writing {}: {e}", path.display())),
        None => std::io::stdout()
            .write_all(bytes)
            .map_err(|e| format!("Error writing stdout: {e}")),
    }
}
