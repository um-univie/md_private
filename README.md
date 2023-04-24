# MD Simulation Suite

**MD Simulation Suite** is a multi-purpose molecular dynamics (MD) simulation tool written in pure Rust. It is designed to efficiently identify bonds and offers a modular approach for various calculations in MD simulations.

## Features

- Efficient bond identification
- Modular design for better flexibility in calculations
- Angle calculations
- Pure Rust implementation for enhanced performance

## Performance Considerations

While the suite offers excellent performance in bond identification, certain calculations such as angle calculations have been designed with a focus on modularity rather than raw performance. This trade-off allows for better flexibility in extending the functionality of the suite.

## Installation

Precompiled binaries for Linux_x86_64 and AARCH64 can be found under release

To compile the **MD Simulation Suite** from source, follow these steps:

1. Install Rust

Unix-like:
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Windows:
Under Standalone installers: https://forge.rust-lang.org/infra/other-installation-methods.html

2. Clone the repository:

```
git clone https://github.com/um-univie/md
```

3. Change to the project directory:

```
cd md
```

4. Build the project:

```
cargo build --release
```


## Precompiled Binaries

The **MD Simulation Suite** has been developed on the ARM architecture. Nonetheless precompiled binaries for all major platforms can be found in the [Releases](https://github.com/your_username/md_simulation_suite/releases) section of the GitHub repository.

## Usage

Run the compiled binary with the appropriate command and input file or folder:

- To render a file:
```
./md render /path/to/input/files
```

- To analyze a file or a batch of files in a folder:
```
./md analyze /path/to/input/file_or_folder
```
## License

This project is licensed under the [MIT License](LICENSE).


