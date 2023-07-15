use std::collections::HashSet;
use std::path::PathBuf;

#[derive(Debug)]
struct IgnoreMacros(HashSet<String>);

impl bindgen::callbacks::ParseCallbacks for IgnoreMacros {
    fn will_parse_macro(&self, name: &str) -> bindgen::callbacks::MacroParsingBehavior {
        if self.0.contains(name) {
            bindgen::callbacks::MacroParsingBehavior::Ignore
        } else {
            bindgen::callbacks::MacroParsingBehavior::Default
        }
    }
}

// https://fitzgeraldnick.com/2016/12/14/using-libbindgen-in-build-rs.html
fn main() {
    let gvc = pkg_config::probe_library("libgvc").unwrap();
    let cgraph = pkg_config::probe_library("libcgraph").unwrap();
    let include_dirs = |lib: pkg_config::Library| {
        lib.include_paths.into_iter().map(|p| format!("-I{}", p.to_string_lossy()))
    };

    // https://github.com/rust-lang/rust-bindgen/issues/687
    let ignored_macros = IgnoreMacros(
        vec![
            "FP_INFINITE".into(),
            "FP_NAN".into(),
            "FP_NORMAL".into(),
            "FP_SUBNORMAL".into(),
            "FP_ZERO".into(),
        ]
        .into_iter()
        .collect(),
    );

    let bindings = bindgen::Builder::default()
        .clang_args(include_dirs(gvc))
        .clang_args(include_dirs(cgraph))
        .header("wrapper.h")
        .parse_callbacks(Box::new(ignored_macros))
        .rustfmt_bindings(true)
        .generate() // Finish the builder and generate the bindings.
        .expect("unable to generate bindings");

    let out_path = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out_path.join("bindings.rs")).expect("cannot write bindings");
}
