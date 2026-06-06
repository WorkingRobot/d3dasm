//! Verify the DXBC checksum implementation against real shaders: recompute the
//! header digest from each container's content and compare to the stored hash.
//!
//! Usage: cargo run --release --example hash_check -- shaders/**/*.bin

fn main() {
    let mut total = 0usize;
    let mut matched = 0usize;

    for arg in std::env::args().skip(1) {
        let data = match std::fs::read(&arg) {
            Ok(d) => d,
            Err(_) => continue,
        };
        for c in dxbc::scan_dxbc(&data) {
            total += 1;
            // Hash covers container bytes from offset 20 to the end.
            let start = c.offset_in_file + 20;
            let end = c.offset_in_file + c.total_size as usize;
            let computed = dxbc::checksum::dxbc_checksum(&data[start..end]);
            if computed == c.header_hash {
                matched += 1;
            } else if total - matched <= 3 {
                eprintln!(
                    "MISMATCH {arg} @0x{:X}\n  stored:   {}\n  computed: {}",
                    c.offset_in_file,
                    hex(&c.header_hash),
                    hex(&computed)
                );
            }
        }
    }

    println!("{matched}/{total} container hashes reproduced");
    if matched != total {
        std::process::exit(1);
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
