extern crate md;
use md::*;
mod renderer;
use std::path::Path;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <render|analyze> <input_path>", args[0]);
        return;
    }

    let command = &args[1];
    let input_path = &args[2];
    match command.as_str() {
        "render" => {
            println!("Rendering file: {}", input_path);
            render(input_path)
        }
        "analyze" => {
            if Path::new(input_path).is_dir() {
                println!("Analyzing files in folder: {}", input_path);
                analyze(input_path, true);
            } else {
                println!("Analyzing file: {}", input_path);
                analyze(input_path, false);
            }
        }
        _ => {
            eprintln!(
                "Invalid command. Usage: {} <render|analyze> <input_path>",
                args[0]
            );
        }
    }
}
