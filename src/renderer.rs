use super::MolecularSystem;
use bevy::{prelude::*, window::PresentMode};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

#[derive(Component)]
struct Shape;

pub fn render(pdb_path: &str) {
    let mut molecularsystem = MolecularSystem::from_pdb(pdb_path);
    molecularsystem.find_bonds(2.0);
    run(molecularsystem);
}

pub fn run(molecules: MolecularSystem) {
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "MD Simulation Suite".into(),
                present_mode: PresentMode::AutoVsync,
                fit_canvas_to_parent: true,
                prevent_default_event_handling: false,
                ..default()
            }),
            ..default()
        }))
        .add_plugin(PanOrbitCameraPlugin)
        .insert_resource(molecules)
        .add_startup_system(setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    molecularsystem: Res<MolecularSystem>,
) {
    let material = materials.add(StandardMaterial::default());
    let shape = meshes.add(Mesh::from(shape::UVSphere {
        radius: 0.3,
        sectors: 20,
        ..default()
    }));

    let cylinder = meshes.add(
        shape::Cylinder {
            radius: 0.1,
            height: 1.4,
            ..Default::default()
        }
        .into(),
    );
    for molecule in &molecularsystem.molecules {
        for atom in &molecule.atoms {
            commands.spawn(PbrBundle {
                mesh: shape.clone(),
                material: material.clone(),
                transform: Transform::from_translation(atom.position.to_vec3()),
                ..default()
            });
        }
        for bond in &molecule.bonds {
            let transform = molecule.atoms[bond.0]
                .position
                .vector_rotation_transform(&molecule.atoms[bond.1].position);
            commands.spawn(PbrBundle {
                mesh: cylinder.clone(),
                material: material.clone(),
                transform,
                ..default()
            });
        }
    }
    let center = molecularsystem.center().to_vec3();
    commands.spawn((
        Camera3dBundle::default(),
        PanOrbitCamera {
            focus: center,
            ..default()
        },
    ));
}
