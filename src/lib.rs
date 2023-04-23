mod renderer;
use bevy::prelude::*;
use csv::Writer;
use kdtree::distance::squared_euclidean;
use kdtree::KdTree;
use ndarray::{s, Array2, Axis};
use phf::phf_map;
use rand::Rng;
use rand_distr::Gamma;
use rayon::prelude::*;
use regex::Regex;
use std::io::{self, BufRead};
use std::num::ParseFloatError;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Sub};
use std::path::Path;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
};

// Perfect hash function for the standard atomic_weight, based on the NIST dataset: https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=&isotype=some
static ATOMIC_MASSES: phf::Map<&'static str, f32> = phf_map! {
    "H" => 1.008,
    "HE" => 4.0026,
    "LI" => 6.94,
    "BE" => 9.0122,
    "B" => 10.81,
    "C" => 12.011,
    "N" => 14.007,
    "O" => 15.999,
    "F" => 18.998,
    "NE" => 20.180,
    "NA" => 22.990,
    "MG" => 24.305,
    "AL" => 26.982,
    "SI" => 28.085,
    "P" => 30.974,
    "S" => 32.06,
    "CL" => 35.45,
    "AR" => 39.948,
    "K" => 39.098,
    "CA" => 40.078,
    "SC" => 44.956,
    "TI" => 47.867,
    "V" => 50.942,
    "CR" => 51.996,
    "MN" => 54.938,
    "FE" => 55.845,
    "CO" => 58.933,
    "NI" => 58.693,
    "CU" => 63.546,
    "ZN" => 65.38,
    "GA" => 69.723,
    "GE" => 72.630,
    "AS" => 74.922,
    "SE" => 78.971,
    "BR" => 79.904,
    "KR" => 83.798,
    "RB" => 85.4678,
    "SR" => 87.62,
    "Y" => 88.90584,
    "ZR" => 91.224,
    "NB" => 92.90637,
    "MO" => 95.95,
    "TC" => 98.0,
    "RU" => 101.07,
    "RH" => 102.90549,
    "PD" => 106.42,
    "AG" => 107.8682,
    "CD" => 112.414,
    "IN" => 114.818,
    "SN" => 118.710,
    "SB" => 121.760,
    "TE" => 127.60,
    "I" => 126.90447,
    "XE" => 131.293,
};

// Only single bonds are condsidered, and the values are taken from: https://chemistry-europe.onlinelibrary.wiley.com/doi/10.1002/chem.200800987
static ATOMIC_RADII: phf::Map<&'static str, f32> = phf_map! {
    "H" => 0.32,
    "HE" => 0.46,
    "LI" => 1.33,
    "BE" => 1.02,
    "B" => 0.85,
    "C" => 0.75,
    "N" => 0.75,
    "O" => 0.73,
    "F" => 0.72,
    "NE" => 0.58,
    "NA" => 1.60,
    "MG" => 1.39,
    "AL" => 1.26,
    "SI" => 1.16,
    "P" => 1.11,
    "S" => 1.03,
    "CL" => 0.99,
    "AR" => 0.97,
    "K" => 2.03,
    "CA" => 1.74,
    "SC" => 1.44,
    "TI" => 1.32,
    "V" => 1.22,
    "CR" => 1.18,
    "MN" => 1.17,
    "FE" => 1.17,
    "CO" => 1.16,
    "NI" => 1.15,
    "CU" => 1.17,
    "ZN" => 1.25,
    "GA" => 1.26,
    "GE" => 1.22,
    "AS" => 1.21,
    "SE" => 1.16,
    "BR" => 1.14,
    "KR" => 1.12,
    "RB" => 2.16,
    "SR" => 1.91,
    "Y" => 1.62,
    "ZR" => 1.45,
    "NB" => 1.34,
    "MO" => 1.29,
    "TC" => 1.27,
    "RU" => 1.25,
    "RH" => 1.25,
    "PD" => 1.20,
    "AG" => 1.39,
    "CD" => 1.44,
    "IN" => 1.42,
    "SN" => 1.39,
    "SB" => 1.39,
    "TE" => 1.38,
    "I" => 1.39,
    "XE" => 1.40,
};

/// A 3D vector represented by its x, y, and z components.
///
/// # Example
///
/// ```
/// use md::Vector;
///
/// fn main() {
///     let v1 = Vector::new(1.0, 2.0, 3.0);
///     let v2 = Vector::new(4.0, 5.0, 6.0);
///
///     let v3 = v1 + v2;
///     let v4 = v1 * 2.0;
///     let v5 = v1 / 2.0;
///     println!("Vector sum: {:?}", v3);
///     println!("Vector scalar multiplication: {:?}", v3);
///     println!("Vector scalar division: {:?}", v3);
/// }
/// ```
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct Vector {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, Component)]
pub struct Atom {
    pub name: String,
    pub atom_type: String,
    pub position: Vector,
    pub velocity_vector: Vector,
}

/// This function   
#[derive(Debug, Resource)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<(usize, usize)>,
    pub bond_angles: Vec<((usize, usize, usize), f32)>,
    pub dihedral_angles: Vec<((usize, usize, usize, usize), f32)>,
}
#[derive(Debug, Resource)]
pub struct MolecularSystem {
    pub molecules: Vec<Molecule>,
}

// Implement the Add trait for Vector
impl Add<Vector> for Vector {
    type Output = Vector;

    fn add(self, other: Vector) -> Vector {
        Vector {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}

// Implement the Mul trait for Vector
impl Mul<f32> for Vector {
    type Output = Vector;

    fn mul(self, scalar: f32) -> Vector {
        Vector {
            x: self.x * scalar,
            y: self.y * scalar,
            z: self.z * scalar,
        }
    }
}

// Implement the Div trait for Vector
impl Div<f32> for Vector {
    type Output = Vector;

    fn div(self, scalar: f32) -> Self::Output {
        Vector {
            x: self.x / scalar,
            y: self.y / scalar,
            z: self.z / scalar,
        }
    }
}
// Implement the Sub trait for Vector
impl Sub<Vector> for Vector {
    type Output = Vector;

    fn sub(self, rhs: Vector) -> Self::Output {
        Vector::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

// Implement the AddAssign trait for Vector
impl AddAssign for Vector {
    fn add_assign(&mut self, other: Self) {
        self.x += other.x;
        self.y += other.y;
        self.z += other.z;
    }
}

// Implement the AddAssign trait for Vector
impl MulAssign<f32> for Vector {
    fn mul_assign(&mut self, scalar: f32) {
        self.x = self.x * scalar;
        self.y = self.y * scalar;
        self.z = self.z * scalar;
    }
}

impl Vector {
    /// Creates a default `Vector` being the zero vector
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use md::Vector;
    ///
    /// fn main() {
    ///     let v = Vector::default();
    ///     println!("Vector: {:?}", v);
    ///     assert_eq!(v.length(),0.0)
    /// }
    /// ```
    ///
    pub fn default() -> Vector {
        Vector::new(0.0, 0.0, 0.0)
    }
    /// Creates a `Vector` from an array of 3 elements, representing the x, y, and z components.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use md::Vector;
    ///
    /// fn main() {
    ///     let array = [1.0, 2.0, 3.0];
    ///     let v = Vector::from_vec(&array);
    ///     println!("Vector: {:?}", v);
    /// }
    /// ```
    ///
    pub fn from_vec(array: &[f32; 3]) -> Vector {
        Vector {
            x: array[0],
            y: array[1],
            z: array[2],
        }
    }
    /// Creates a `Vector` from x, y, and z components.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use md::Vector;
    ///
    /// fn main() {
    ///     let v = Vector::new(1.0, 2.0, 3.0);
    ///     println!("Vector: {:?}", v);
    /// }
    /// ```
    ///
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    pub fn x() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }
    pub fn y() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }
    pub fn z() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }
    /// Calculates the difference between two `Vector` objects.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use md::Vector;
    ///
    /// fn main() {
    ///     let v1 = Vector::new(1.0, 2.0, 3.0);
    ///     let v2 = Vector::new(4.0, 5.0, 6.0);
    ///
    ///     let v_diff = v1.difference(&v2);
    ///     println!("Vector difference: {:?}", v_diff);
    /// }
    /// ```
    pub fn difference(&self, other: &Self) -> Self {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dz = self.z - other.z;
        Self {
            x: dx,
            y: dy,
            z: dz,
        }
    }
    /// Calculates the length of a vector.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use md::Vector;
    ///
    /// fn main() {
    ///     let v = Vector::new(1.0, 1.0, 1.0);
    ///     println!("Vector: {:?}", v.length());
    ///     assert_eq!(v.length(), 3.0_f32.sqrt())
    /// }
    ///
    /// ```
    pub fn length(&self) -> f32 {
        (self.x.powi(2) + self.y.powi(2) + self.z.powi(2)).sqrt()
    }
    /// Calculates the dot product of two vectors.
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use md::Vector;
    ///
    /// fn main() {
    ///     let v1 = Vector::new(1.0, 2.0, 3.0);
    ///     let v2 = Vector::new(4.0, 5.0, 6.0);
    ///     println!("Vector: {:?}", v1.dot(&v2));
    /// }
    ///
    /// ```
    pub fn dot(&self, other: &Self) -> f32 {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    /// Calculates the distance between to vectors
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use md::Vector;
    ///
    /// fn main() {
    ///     let v1 = Vector::new(1.0, 2.0, 3.0);
    ///     let v2 = Vector::new(4.0, 5.0, 6.0);
    ///     println!("Vector: {:?}", v1.distance(&v2));
    /// }
    ///
    /// ```
    pub fn distance(&self, other: &Self) -> f32 {
        self.difference(&other).length()
    }
    /// Documentation
    pub fn distance_squared(&self, other: &Self) -> f32 {
        self.difference(&other).x.powi(2)
            + self.difference(&other).y.powi(2)
            + self.difference(&other).z.powi(2)
    }
    /// Generates a random unit vector
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use md::Vector;
    ///
    /// fn main() {
    ///     let v = Vector::random_unit_vector();
    ///     println!("Vector: {:?}", v);
    ///     assert_eq!(v.length().ceil(),1.0)
    /// }
    ///
    /// ```
    ///
    pub fn random_unit_vector() -> Self {
        let mut rng = rand::thread_rng();
        let theta = rng.gen_range(0.0..std::f32::consts::PI);
        let phi = rng.gen_range(0.0..2.0 * std::f32::consts::PI);

        let x = theta.sin() * phi.cos();
        let y = theta.sin() * phi.sin();
        let z = theta.cos();

        Vector::new(x, y, z)
    }
    /// Calculates the cross product of two vectors and returns a new vector
    ///
    /// # Panics
    ///
    /// This function does not panic.
    ///
    /// # Examples
    ///
    /// ```
    /// use md::Vector;
    ///
    /// fn main() {
    ///     let vec1 = Vector::x();
    ///     let vec2 = Vector::y();
    ///     assert_eq!(vec1.cross(&vec2),Vector::z())
    /// }
    ///
    /// ```
    pub fn cross(&self, other: &Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }
    /// Returns the angle between two vectors in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use md::Vector;
    ///
    /// let vec_a = Vector::new(1.0, 0.0, 0.0);
    /// let vec_b = Vector::new(0.0, 1.0, 0.0);
    ///
    /// let angle = vec_a.angle_between(&vec_b);
    /// assert_eq!(angle, std::f32::consts::FRAC_PI_2);
    /// ```
    pub fn angle_between(&self, other: &Vector) -> f32 {
        let lengths_product = self.length() * other.length();
        if lengths_product == 0.0 {
            0.0
        } else {
            let angle_cosine = (&self.dot(other) / (lengths_product)).clamp(-1.0, 1.0);
            let angle = angle_cosine.acos();
            angle
        }
    }
    /// Calculates a normal vector for a given input vector
    ///
    /// # Panics
    /// This function does not panic as the case of a zero length vector is handeled.
    ///
    /// # Example
    ///
    /// ```
    /// use md::Vector;
    ///
    /// fn main() {
    ///     let vec1 = Vector::new(1.0,2.0,3.0);
    ///     assert_eq!(vec1.normalize().length().ceil(),1.0)
    /// }
    ///
    pub fn normalize(&self) -> Self {
        if self.length() == 0.0 {
            *self
        } else {
            *self / self.length()
        }
    }
    /// Calculates the angle given three points in the order, start, middle, end
    ///
    /// # Panics
    /// Function does not panic
    ///
    /// # Example
    /// ```
    /// use md::Vector;
    ///
    /// fn main() {
    ///     let start_point = Vector::new(1.0,0.0,0.0);
    ///     let middle_point = Vector::new(0.0,0.0,0.0);
    ///     let end_point = Vector::new(0.0,1.0,0.0);
    ///     assert_eq!(start_point.angle_between_points(&middle_point,&end_point),1.5707964);
    /// }
    /// ```
    pub fn angle_between_points(&self, middle_point: &Vector, end_point: &Vector) -> f32 {
        let v1 = *middle_point - *self;
        let v2 = *end_point - *middle_point;
        v1.angle_between(&v2)
    }
    pub fn vector_rotation_transform(&self, vector_b: &Vector) -> Transform {
        let direction = *vector_b - *self;

        let y_unit_vector = Vector::y();
        let axis = y_unit_vector.cross(&direction).normalize();
        let angle = y_unit_vector.angle_between(&direction);
        let rotation = Quat::from_axis_angle(axis.to_vec3(), angle);
        let position = *self + direction / 2.0;

        let transform = Transform {
            translation: position.to_vec3(),
            rotation,
            scale: Vec3::ONE,
        };

        transform
    }
    pub fn to_tuple(&self) -> (f32, f32, f32) {
        (self.x, self.y, self.z)
    }
    pub fn to_vec3(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }
    pub fn to_array(&self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
    pub fn to_vec(&self) -> Vec<f32> {
        vec![self.x, self.y, self.z]
    }
}

impl Atom {
    pub fn default() -> Atom {
        Atom {
            name: "C".to_string(),
            atom_type: "Ca".to_string(),
            position: Vector::default(),
            velocity_vector: Vector::default(),
        }
    }
    /// Calculates the kinetic energy of an atom.
    ///
    /// # Panics
    /// Function does not panic.
    ///
    /// # Example
    ///
    /// ```
    /// use md::Atom;
    ///
    /// fn main() {
    ///     let temperature = 200.0; // Kelvin
    ///     let mut atom1 = Atom::default();
    ///     atom1.random_velocity_vector(temperature);
    ///     println!("{}",atom1.kinetic_energy())
    ///    }
    pub fn kinetic_energy(&self) -> f32 {
        0.5 * self.mass() * self.velocity_vector.length().powi(2)
    }
    /// Calculates the distance between two atoms.
    ///
    /// # Parameters
    /// * 'self' - A reference to the current atom.
    /// * 'other' - A reference to the other atom.
    ///
    /// # Returns
    /// The distance between the two atoms as a 64-bit floating-point number.
    pub fn distance(&self, other: &Atom) -> f32 {
        self.position.distance(&other.position)
    }
    /// Calculates the potential energy of a diatomic system using the harmonic oscillator model.
    ///
    /// # Parameters
    /// * 'other' - A reference to the other atom.
    /// * 'force_constant' - The force constant (k) of the diatomic system.
    /// * 'equilibrium_distance' - The equilibrium distance between the two atoms.
    ///
    /// # Returns
    /// The potential energy of the diatomic system as a 64-bit floating-point number.
    pub fn potential_energy(
        &self,
        other: &Atom,
        force_constant: f32,
        equilibrium_distance: f32,
    ) -> f32 {
        0.5 * force_constant * (self.distance(&other) - equilibrium_distance).powi(2)
    }
    /// Calculates the Hamiltonian for two atoms, considering their kinetic energies and potential energy.
    ///
    /// The Hamiltonian is the sum of the kinetic energies of the atoms and their potential energy.
    /// The potential energy is calculated using a force constant and equilibrium distance.
    ///
    /// # Panics
    ///
    /// Panics when atom is not valid
    ///
    /// # Examples
    ///
    /// ```
    /// use md::{Atom, Vector};
    ///
    /// fn main() {
    ///     let atom1 = Atom {position: Vector::new(0.0, 0.0, 0.0), velocity_vector: Vector::new(1.0, 1.0, 1.0), ..Atom::default()};
    ///     let atom2 = Atom {position: Vector::new(1.0, 1.0, 1.0), velocity_vector: Vector::new(-1.0, -1.0, -1.0), ..Atom::default()};
    ///
    ///     let force_constant = 100.0; // N/m
    ///     let equilibrium_distance = 0.1; // m
    ///
    ///     let hamiltonian_value = atom1.hamiltonian(&atom2, force_constant, equilibrium_distance);
    ///     println!("Hamiltonian value: {}", hamiltonian_value);
    /// }
    /// ```
    pub fn hamiltonian(&self, other: &Atom, force_constant: f32, equilibrium_distance: f32) -> f32 {
        self.kinetic_energy()
            + other.kinetic_energy()
            + self.potential_energy(&other, force_constant, equilibrium_distance)
    }
    pub fn mass(&self) -> f32 {
        *ATOMIC_MASSES.get(&self.name).unwrap()
    }
    pub fn get_atomic_radius(&self) -> f32 {
        *ATOMIC_RADII.get(&self.name).unwrap()
    }
    pub fn random_velocity_vector(&mut self, temperature: f32) {
        let mass_kg = self.mass() * 1.66054e-27; // Conversion from amu to kg
                                                 // Boltzmann constant in J/K
        let k_b = 1.38065e-23;
        let sigma = (k_b * temperature / mass_kg).sqrt();
        // Calculate the Gamma distribution parameters
        let degrees_of_freedom = 3.0;
        let shape = degrees_of_freedom / 2.0;
        let scale = 2.0 * sigma.powi(2);
        let gamma = Gamma::new(shape, scale).unwrap();
        let mut rng = rand::thread_rng();
        let velocity = rng.sample(&gamma);
        self.velocity_vector = Vector::random_unit_vector() * velocity
    }
}

impl MolecularSystem {
    pub fn from_pdb<P: AsRef<Path>>(path: P) -> MolecularSystem {
        let file = File::open(&path).expect("Could not open file");
        let reader = io::BufReader::new(&file);

        let mut atom_groups: Vec<Vec<String>> = vec![Vec::new()];

        for line in reader.lines() {
            if let Ok(line) = line {
                if line.starts_with("ATOM") || line.starts_with("HETATM") {
                    atom_groups.last_mut().unwrap().push(line);
                } else if line.starts_with("TER") {
                    atom_groups.push(Vec::new());
                } else if line.starts_with("ENDMDL") {
                    break;
                } else {
                    continue;
                }
            }
        }
        let path = path.as_ref().to_str().unwrap();
        if path.ends_with("pdb") {
            MolecularSystem {
                molecules: atom_groups
                    .into_iter()
                    .map(|group| {
                        let atoms: Vec<Atom> = group
                            .par_iter()
                            .map(|line| extract_atom_pdb(line).expect("Could not parse Atom"))
                            .collect();
                        Molecule {
                            atoms,
                            bonds: Vec::new(),
                            bond_angles: Vec::new(),
                            dihedral_angles: Vec::new(),
                        }
                    })
                    .collect(),
            }
        } else if path.ends_with("cif") {
            MolecularSystem {
                molecules: atom_groups
                    .into_iter()
                    .map(|group| {
                        let atoms: Vec<Atom> = group
                            .par_iter()
                            .map(|line| extract_atom_cif(line).expect("Could not parse Atom"))
                            .collect();
                        Molecule {
                            atoms,
                            bonds: Vec::new(),
                            bond_angles: Vec::new(),
                            dihedral_angles: Vec::new(),
                        }
                    })
                    .collect(),
            }
        } else {
            MolecularSystem {
                molecules: vec![Molecule::default()],
            }
        }
    }

    pub fn center(&self) -> Vector {
        let mut sum = Vector::new(0.0, 0.0, 0.0);
        let mut count = 0.0;
        for molecule in &self.molecules {
            for atom in &molecule.atoms {
                sum += atom.position
            }
            count += molecule.atoms.len() as f32
        }
        sum / count
    }
    pub fn find_bonds(&mut self, threshold: f32) {
        self.molecules
            .par_iter_mut()
            .for_each(|molecule| molecule.find_bonds(threshold))
    }
    pub fn number_of_atoms(&self) -> usize {
        self.molecules
            .iter()
            .map(|molecule| molecule.atoms.len())
            .sum()
    }
    pub fn number_of_bonds(&self) -> usize {
        self.molecules
            .iter()
            .map(|molecule| molecule.bonds.len())
            .sum()
    }
    pub fn number_of_bond_angles(&self) -> usize {
        self.molecules
            .iter()
            .map(|molecule| molecule.bond_angles.len())
            .sum()
    }
    pub fn number_of_dihedral_angles(&self) -> usize {
        self.molecules
            .iter()
            .map(|molecule| molecule.dihedral_angles.len())
            .sum()
    }
    pub fn identify_angles(&mut self) {
        self.molecules
            .par_iter_mut()
            .for_each(|molecule| molecule.identify_angles())
    }
    pub fn identify_dihedrals(&mut self) {
        self.molecules
            .par_iter_mut()
            .for_each(|molecule| molecule.identify_dihedrals())
    }
}

impl Molecule {
    pub fn default() -> Molecule {
        Molecule {
            atoms: vec![Atom::default()],
            bonds: vec![],
            bond_angles: vec![],
            dihedral_angles: vec![],
        }
    }
    /// Creates a new molecule from a list of atoms
    ///
    /// # Panics
    /// Function panics if the list of atoms is empty or if the file cannot be opened
    ///
    /// # Examples
    /// ```
    /// use md::*;
    ///
    /// fn main() {
    ///    let molecule = Molecule::from_xyz("tests/ethane.xyz");
    ///     assert_eq!(molecule.atoms.len(), 8);
    /// }
    /// ```
    pub fn from_xyz(filepath: &str) -> Molecule {
        let file = File::open(filepath).expect("Could not open file");
        let reader = io::BufReader::new(file);
        let re = Regex::new(r"([A-Za-z]+)\s+([\d\.-]+)\s+([\d\.-]+)\s+([\d\.-]+)").unwrap();
        let atoms = reader
            .lines()
            .skip(2)
            .filter_map(|line| {
                if let Ok(line) = line {
                    let caps = re.captures(&line).unwrap();
                    let name = caps.get(1).unwrap().as_str().to_string();
                    let x = caps.get(2).unwrap().as_str().parse::<f32>().unwrap();
                    let y = caps.get(3).unwrap().as_str().parse::<f32>().unwrap();
                    let z = caps.get(4).unwrap().as_str().parse::<f32>().unwrap();
                    Some(Atom {
                        name,
                        position: Vector::new(x, y, z),
                        ..Atom::default()
                    })
                } else {
                    None
                }
            })
            .collect::<Vec<Atom>>();
        Molecule {
            atoms,
            bonds: vec![],
            bond_angles: vec![],
            dihedral_angles: vec![],
        }
    }
    pub fn find_bonds_from_distance_matrix(&mut self) {
        let distance_matrix = self.distance_matrix();
        let bonds: Vec<(usize, usize)> = distance_matrix
            .axis_iter(Axis(0))
            .into_par_iter()
            .enumerate()
            .map(|(i, row)| {
                row.slice(s![i..])
                    .iter()
                    .enumerate()
                    .filter_map(|(j, distance)| {
                        if i < j
                            && *distance
                                > self.atoms[i].get_atomic_radius()
                                    + self.atoms[j].get_atomic_radius()
                        {
                            Some((i, j))
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<(usize, usize)>>()
            })
            .flatten()
            .collect();
        self.bonds = bonds;
    }
    pub fn find_bonds(&mut self, threshold: f32) {
        let mut kdtree = KdTree::new(3);
        for (i, atom) in self.atoms.iter().enumerate() {
            kdtree
                .add([atom.position.x, atom.position.y, atom.position.z], i)
                .unwrap();
        }
        let threshold_squared = threshold.powi(2);
        let bonds = self
            .atoms
            .par_iter()
            .enumerate()
            .map(|(atom_index, atom)| {
                let neighbors = kdtree
                    .within(
                        &[atom.position.x, atom.position.y, atom.position.z],
                        threshold_squared,
                        &squared_euclidean,
                    )
                    .expect("Problem with the kdtree search");
                let mut bonds = Vec::with_capacity(neighbors.len());
                for (distance, neighbor_atom_index) in neighbors {
                    if atom_index < *neighbor_atom_index {
                        let neighbor_atom = &self.atoms[*neighbor_atom_index];
                        let bond_distance = *ATOMIC_RADII
                            .get(&atom.name)
                            .expect(&format!("Atom type not found: {}", atom.name))
                            + *ATOMIC_RADII
                                .get(&neighbor_atom.name)
                                .expect(&format!("Atom type not found: {}", neighbor_atom.name));
                        // adding a safety margin of 20 % here to not miss any bonds
                        let bond_distance_squared = (bond_distance * 1.2).powi(2);
                        if distance <= bond_distance_squared {
                            bonds.push((atom_index, *neighbor_atom_index))
                        }
                    }
                }
                bonds
            })
            .flatten()
            .collect();
        self.bonds = bonds;
    }

    pub fn center(&self) -> Vector {
        let mut sum = Vector::new(0.0, 0.0, 0.0);

        for atom in &self.atoms {
            sum += atom.position
        }

        let count = self.atoms.len() as f32;
        sum / count
    }
    /// This function calculates and assigns random velocities based on the maxwell boltzmann distribution to each atom in a Molecule
    ///
    /// Comment: this function needs some serious optimization as there are a lot of distributions generated which could be shared between atoms of the same type. The loop is also not shareable efficiently without removal of the iter_mut() as it would necessitate Arc(Mutex())
    ///
    /// # Panics
    /// This function does not panic but it throws and error and default to a different temperature if the temperature is negative
    ///
    /// # Example
    /// ```
    /// use md::Molecule;
    ///
    /// fn main() {
    ///     let mut molecule = Molecule::default();
    ///     let temperature = 298.0; // K
    ///     molecule.random_velocities(temperature);
    ///     println!("{:?}",molecule)
    /// }
    /// ```
    pub fn random_velocities(&mut self, temperature: f32) {
        // Error handling
        let mut temperature = temperature;
        if temperature <= 0.0 {
            println!("Please input a temperature over 0 K, defaulting to 298 K");
            temperature = 298.0;
        }
        // Generate speeds for each atom
        self.atoms
            .par_iter_mut()
            .for_each(|atom| atom.random_velocity_vector(temperature));
    }
    /// This function computes the upper triangular distance matrix for a given molecule
    ///
    /// # Panics
    ///
    /// This function does not panic
    ///
    /// # Example
    ///
    /// ```
    /// use md::{Molecule,Vector,Atom};
    ///
    /// fn main() {
    ///     let molecule = Molecule {
    ///                             atoms: vec![Atom {position: Vector::new(0.0,0.0,0.0),..Atom::default()},Atom{position: Vector::new(0.0,1.0,0.0),..Atom::default()},Atom{position: Vector::new(1.0,0.0,0.0),..Atom::default()}],
    ///                             ..Molecule::default()};
    ///     let distance_matrix = molecule.distance_matrix();
    ///     println!("{}",distance_matrix)
    /// }
    /// ```
    pub fn distance_matrix(&self) -> Array2<f32> {
        let length = self.atoms.len();
        let mut matrix = Array2::<f32>::zeros((length, length));
        matrix
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(mut index, mut row)| {
                for element in row.slice_mut(s![index..]) {
                    *element = self.atoms[index].distance(&self.atoms[index]);
                    index += 1;
                }
            });
        matrix
    }
    //not getting all angles
    pub fn identify_angles(&mut self) {
        if self.bonds.is_empty() {
            self.find_bonds(2.0)
        }
        let mut angles = Vec::with_capacity(self.atoms.len() * 2);

        for &(atom1, atom2) in self.bonds.iter() {
            for &(atom3, atom4) in self.bonds.iter() {
                if atom1 == atom3 && atom2 == atom4 {
                    continue;
                }
                if let Some((a, b, c)) = match (atom1, atom2, atom3, atom4) {
                    (a, b, c, d) if a == c => Some((b, a, d)),
                    (a, b, c, d) if a == d => Some((b, a, c)),
                    (a, b, c, d) if b == c => Some((a, b, d)),
                    (a, b, c, d) if c == d => Some((a, c, b)),
                    _ => None,
                } {
                    if c > a {
                        let point1 = &self.atoms[a].position;
                        let point2 = &self.atoms[b].position;
                        let point3 = &self.atoms[c].position;
                        let angle = point1.angle_between_points(point2, point3);
                        angles.push(((a, b, c), angle))
                    }
                }
            }
        }
        self.bond_angles = angles;
    }

    /// This function identifies all dihedral angles in a molecule
    ///
    /// # Panics
    /// This function does not panic
    ///
    /// # Example
    ///
    /// ```
    ///use md::*;
    ///
    ///fn main() {
    ///    let mut molecule = Molecule::from_xyz("tests/ethane.xyz");
    ///
    ///    molecule.identify_dihedrals();
    ///
    ///    assert_eq!(molecule.dihedral_angles.len(), 9, "Incorrect number of dihedral angles identified");
    ///    assert_eq!(molecule.dihedral_angles[0], ((3, 0, 1, 7), 1.0472065), "Incorrect dihedral angle identified");
    ///}
    ///```
    pub fn identify_dihedrals(&mut self) {
        if self.bond_angles.is_empty() {
            self.identify_angles()
        }
        let mut dihedrals = Vec::with_capacity(self.bond_angles.len());

        for (angle, _) in &self.bond_angles {
            for &(bond0, bond1) in &self.bonds {
                let (a, b, c) = *angle;
                if bond0 == b || bond1 == b {
                    continue;
                }
                let (b1, b2, b3, b4) = match (a, c) {
                    _ if a == bond0 => (bond1, bond0, b, c),
                    _ if a == bond1 => (bond0, bond1, b, c),
                    _ if c == bond0 => (a, b, bond1, bond0),
                    _ if c == bond1 => (a, b, bond0, bond1),
                    _ => continue,
                };

                if b1 < b4 {
                    dihedrals.push(((b1, b2, b3, b4), self.dihedral_angle(&(b1, b2, b3, b4))));
                }
            }
        }
        self.dihedral_angles = dihedrals;
    }
    /// This function calculates the dihedral angle for all atoms
    ///
    /// # Panics
    /// This function does not panic.
    ///
    /// # Example
    ///
    /// ```
    /// use md::*;
    /// fn main() {
    ///     let mut molecule = Molecule::from_xyz("tests/ethane.xyz");
    ///     molecule.identify_dihedrals();
    ///     assert_eq!(molecule.dihedral_angles[0].1, 1.0472065, "Incorrect dihedral angle identified");
    /// }
    pub fn dihedral_angle(&self, dihedral: &(usize, usize, usize, usize)) -> f32 {
        let (a, b, c, d) = (
            self.atoms[dihedral.0].position,
            self.atoms[dihedral.1].position,
            self.atoms[dihedral.2].position,
            self.atoms[dihedral.3].position,
        );
        let v1 = b - a;
        let v2 = c - b;
        let v3 = d - c;
        let normal1 = v1.cross(&v2);
        let normal2 = v2.cross(&v3);

        let angle = normal1.angle_between(&normal2);
        let sign = normal1.cross(&normal2).dot(&v2);
        if sign < 0.0 {
            -angle
        } else {
            angle
        }
    }
}
/// This function reads a pdb file line and extracts the atom information
///
/// # Panics
/// This function returns a ParseFloatError if the line has errors in the float region, it does not check for other errors.
///
/// ```
/// use md::*;
///
/// fn main() {
///     let line =  "ATOM   2073  CB  ALA B 128      11.390 -11.172  71.797  1.00 16.79           C";
///     let atom = extract_atom_pdb(line).unwrap();
///     assert_eq!(atom.position.x, 11.390, "Incorrect x coordinate");
///     assert_eq!(atom.position.y, -11.172, "Incorrect y coordinate");
///     assert_eq!(atom.position.z, 71.797, "Incorrect z coordinate");
///     assert_eq!(atom.name, "C", "Incorrect atom name");
///     assert_eq!(atom.atom_type, "CB", "Incorrect atom type");
/// }
/// ```
pub fn extract_atom_pdb(line: &str) -> Result<Atom, ParseFloatError> {
    let position = Vector {
        x: line[31..=37].trim().parse::<f32>()?,
        y: line[38..=45].trim().parse::<f32>()?,
        z: line[46..=53].trim().parse::<f32>()?,
    };
    let atom_type = line[12..=16].trim().to_string();
    let name = line[76..=77].trim().to_string();
    Ok(Atom {
        position,
        velocity_vector: Vector::default(),
        name,
        atom_type,
    })
}
/// This function reads a cif file line and extracts the atom information
///
/// # Panics
///
/// This function returns a ParseFloatError if the line has errors in the float region, it does not check for other errors.
///
/// # Example
///
/// ```
/// use md::*;
///
/// fn main() {
///     let line = "ATOM   1    N N   . GLN A 1 1   ? 201.310 198.892 131.429 1.00 70.25  ? 1   GLN A N   1";
///     let atom = extract_atom_cif(line).unwrap();
///     assert_eq!(atom.position.x, 201.310, "Incorrect x coordinate");
///     assert_eq!(atom.position.y, 198.892, "Incorrect y coordinate");
///     assert_eq!(atom.position.z, 131.429, "Incorrect z coordinate");
///     assert_eq!(atom.name, "N", "Incorrect atom name");
///     assert_eq!(atom.atom_type, "N", "Incorrect atom type");
/// }
/// ```
pub fn extract_atom_cif(line: &str) -> Result<Atom, ParseFloatError> {
    let fields: Vec<&str> = line.split_whitespace().collect();
    let atom_type = fields[3].to_string();
    let name = fields[2].to_string();
    let position = Vector {
        x: fields[10].parse::<f32>()?,
        y: fields[11].parse::<f32>()?,
        z: fields[12].parse::<f32>()?,
    };
    Ok(Atom {
        position,
        velocity_vector: Vector::default(),
        name,
        atom_type,
    })
}

pub fn render(pdb_path: &str) {
    let mut molecularsystem = MolecularSystem::from_pdb(&pdb_path);
    molecularsystem.find_bonds(2.0);
    renderer::run(molecularsystem);
}

pub fn analyze<P: AsRef<Path>>(path: P, is_dir: bool) {
    let pdb_path = path;
    if is_dir {
        let paths = fs::read_dir(pdb_path).unwrap();
        let mut writer = Writer::from_path("result.csv").unwrap();
        writer
            .write_record([
                "Name",
                "#Molecules",
                "#atoms",
                "Number of bonds",
                "Number of bond angles",
                "Number of dihedral angles",
                "Time total",
            ])
            .unwrap();
        for path in paths {
            let filename = path.unwrap().path();
            if let Some(extension) = filename.extension() {
                if extension != "pdb" && extension != "cif" {
                    continue;
                }
            } else {
                continue;
            };
            println!("Analyzing {}", filename.to_str().unwrap());
            let molecularsystem = analyze_file(&filename);
            let start_time = std::time::Instant::now();

            let time_elapsed5 = start_time.elapsed();
            let number_of_molecules = molecularsystem.molecules.len();
            let number_of_bonds = molecularsystem.number_of_bonds();
            let number_of_atoms = molecularsystem.number_of_atoms() as u32;
            let number_of_bond_angles = molecularsystem.number_of_bond_angles();
            let number_of_dihedral_angles = molecularsystem.number_of_dihedral_angles();

            writer
                .write_record(&[
                    filename.to_str().unwrap(),
                    &number_of_molecules.to_string(),
                    &number_of_atoms.to_string(),
                    &number_of_bonds.to_string(),
                    &number_of_bond_angles.to_string(),
                    &number_of_dihedral_angles.to_string(),
                    &time_elapsed5.as_millis().to_string(),
                ])
                .unwrap();
        }
    } else {
        analyze_file(&pdb_path);
    }
}

pub fn analyze_file<P: AsRef<Path>>(path: P) -> MolecularSystem {
    let pdb_path = path.as_ref();
    let file = File::create(&format!("{}_result.txt", pdb_path.to_str().unwrap())).unwrap();
    let mut writer2 = BufWriter::new(file);
    let mut molecularsystem = MolecularSystem::from_pdb(&pdb_path);
    if molecularsystem
        .molecules
        .last()
        .expect("No molecules in file")
        .atoms
        .len()
        == 0
    {
        molecularsystem.molecules.pop();
    }
    molecularsystem.find_bonds(2.0);
    molecularsystem.identify_angles();
    molecularsystem.identify_dihedrals();

    for molecule in &molecularsystem.molecules {
        writeln!(writer2, "MOL").unwrap();
        for (index, atom) in molecule.atoms.iter().enumerate() {
            let line = format!(
                "ATOM {} {} {} {} {} {}",
                index, atom.name, atom.atom_type, atom.position.x, atom.position.y, atom.position.z
            );
            writeln!(writer2, "{}", line).unwrap();
        }
        for bonds in &molecule.bonds {
            let line = format!("BOND {} {}", bonds.0, bonds.1);
            writeln!(writer2, "{}", line).unwrap();
        }
        for bond_angle in &molecule.bond_angles {
            let line = format!(
                "BOND_ANGLE {} {} {} {}",
                bond_angle.0 .0, bond_angle.0 .1, bond_angle.0 .2, bond_angle.1
            );
            writeln!(writer2, "{}", line).unwrap();
        }
        for dihedral_angle in &molecule.dihedral_angles {
            let line = format!(
                "DIHEDRAL_ANGLE {} {} {} {} {}",
                dihedral_angle.0 .0,
                dihedral_angle.0 .1,
                dihedral_angle.0 .2,
                dihedral_angle.0 .3,
                dihedral_angle.1
            );
            writeln!(writer2, "{}", line).unwrap();
        }
        writeln!(writer2, "ENDMOL").unwrap();
    }
    molecularsystem
}

// Unit tests (Still needs to be improved)
#[cfg(test)]
mod tests {
    #[test]
    fn test_extract_atom() {
        let line =
            "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N  ";
        let atom = super::extract_atom_pdb(line).unwrap();
        assert_eq!(atom.name, "N");
        assert_eq!(atom.position.x, 10.0);
        assert_eq!(atom.position.y, 10.0);
        assert_eq!(atom.position.z, 10.0);
    }
    #[test]
    fn test_vector_angle() {
        let v1 = super::Vector {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let v2 = super::Vector {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        let angle = v1.angle_between(&v2);
        assert_eq!(angle, std::f32::consts::FRAC_PI_2);
    }
    #[test]
    fn test_cross_product() {
        let v1 = super::Vector {
            x: 1.0,
            y: 0.0,
            z: 0.0,
        };
        let v2 = super::Vector {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        };
        let v3 = v1.cross(&v2);
        assert_eq!(v3.x, 0.0);
        assert_eq!(v3.y, 0.0);
        assert_eq!(v3.z, 1.0);
    }
    #[test]
    fn test_bond_angle() {
        use super::*;
        let atom1 = Atom {
            position: Vector {
                x: 0.5,
                y: 0.0,
                z: 0.0,
            },
            ..Atom::default()
        };
        let atom2 = Atom {
            position: Vector {
                x: 0.0,
                y: 0.5,
                z: 0.0,
            },
            ..Atom::default()
        };
        let atom3 = Atom {
            position: Vector {
                x: 0.0,
                y: 0.0,
                z: 0.5,
            },
            ..Atom::default()
        };
        let mut molecule = Molecule {
            atoms: vec![atom1, atom2, atom3],
            ..Molecule::default()
        };
        molecule.find_bonds(2.0);
        molecule.identify_angles();
        println!("{:?}", molecule.bond_angles);
    }
}
