use std::{fs::File, io::Write, path::Path};

use nalgebra::{point, vector, Isometry3, Point3, Quaternion, UnitQuaternion};
use rapier3d::parry::transformation::vhacd::{VHACDParameters, VHACD};

use clap::Parser;
use serde::{Deserialize, Serialize};

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Gltf file path
    #[arg(short, long)]
    gltf_file: String,

    /// Output directory path
    #[arg(short, long)]
    output_directory: Option<String>,

    /// The string to append to created files
    #[arg(short, long)]
    append: Option<String>,

    /// Max # of hulls to generate
    #[arg(short, long, default_value_t = 1024)]
    max_hulls: u32,

    /// Log the output files on creation
    #[arg(short, long, default_value_t = true)]
    log_success: bool,

    /// Output as a single json file
    #[arg(short, long, default_value_t = false)]
    json_only: bool,
}

fn main() {
    let as_string = std::env::current_dir().unwrap().into_os_string();
    let path_out = as_string.into_string().unwrap();

    let args = Args::parse();
    let clone = args.clone();
    convert_and_write(
        &args.gltf_file,
        &args.output_directory.unwrap_or(path_out),
        clone,
    );
}

fn convert_and_write(path_in: &str, path_out: &str, args: Args) {
    let input_path = Path::new(&path_in);
    let output_path = Path::new(&path_out);

    let (gltf, buffers, _) = gltf::import(input_path).unwrap();
    let mut shapes = Vec::new();

    for scene in gltf.scenes() {
        for node in scene.nodes() {
            match node.mesh() {
                Some(m) => {
                    let mut verts: Vec<Point3<f32>> = Vec::new();
                    let mut indices: Vec<u32> = Vec::new();

                    for primitive in m.primitives() {
                        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));
                        if let Some(iter) = reader.read_positions() {
                            for vertex_position in iter {
                                let point: Point3<f32> = point![
                                    vertex_position[0],
                                    vertex_position[1],
                                    vertex_position[2]
                                ];
                                verts.push(point);
                            }
                        }

                        if let Some(iter) = reader.read_indices() {
                            for read_ind in iter.into_u32() {
                                indices.push(read_ind);
                            }
                        }
                    }

                    let mut tris: Vec<[u32; 3]> = Vec::new();
                    for c in indices.chunks(3) {
                        let t = [c[0], c[1], c[2]];
                        tris.push(t);
                    }

                    let mut parts: Vec<(Vec<Point3<f32>>, Vec<[u32; 3]>)> = Vec::new();
                    let mut params = VHACDParameters::default();
                    params.max_convex_hulls = args.max_hulls;

                    // Apply transform
                    let scale_comp = node.transform().decomposed().2;
                    let translation = node.transform().decomposed().0;
                    let rotation = node.transform().decomposed().1;
                    let quat = Quaternion::new(rotation[0], rotation[1], rotation[2], rotation[3]);
                    let eulers = UnitQuaternion::from_quaternion(quat).euler_angles();

                    let iso = Isometry3::new(
                        vector![translation[0], translation[1], translation[2]],
                        vector![eulers.0, eulers.1, eulers.2],
                    );

                    // Apply rotation and position
                    verts = verts.iter().map(|v| iso.transform_point(v)).collect();

                    // Apply scale
                    verts = verts
                        .iter()
                        .map(|v| {
                            point![
                                v.x * scale_comp[0],
                                v.y * scale_comp[1],
                                v.z * scale_comp[2]
                            ]
                        })
                        .collect();

                    let decomp = VHACD::decompose(&params, &verts, &tris, true);

                    for (vertices, indices) in decomp.compute_exact_convex_hulls(&verts, &tris) {
                        parts.push((vertices.clone(), indices.clone()));
                        shapes.push((vertices.clone(), indices.clone())); // I know, I know..
                    }

                    if args.json_only {
                        continue;
                    }

                    for (i, shape) in parts.into_iter().enumerate() {
                        let name = m.name().unwrap_or("New Obj").to_owned() + &i.to_string();
                        let append = &args.clone().append.unwrap_or("-shape".to_owned());
                        match write_mesh_to_obj(output_path, &name, append, shape, &args) {
                            Ok(_) => {}
                            Err(e) => eprintln!("{}", e),
                        }
                    }
                }
                None => {}
            }
        }
    }
    let append = &args.append.clone().unwrap_or("-shape".to_owned());
    let name = input_path
        .file_name()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    match write_meshes_to_json(output_path, &name, append, shapes, &args) {
        Ok(_) => {}
        Err(e) => eprintln!("{}", e),
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct SerdePoint3 {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Serialize, Deserialize, Debug)]
struct SerdeShape {
    points: Vec<SerdePoint3>,
    tris: Vec<[u32; 3]>,
}

#[derive(Serialize, Deserialize, Debug)]
struct ShapeCollection {
    shapes: Vec<SerdeShape>,
}

fn write_meshes_to_json(
    directory: &Path,
    name: &str,
    append: &str,
    shapes: Vec<(Vec<Point3<f32>>, Vec<[u32; 3]>)>,
    args: &Args,
) -> std::io::Result<()> {
    let filename_fmt = format!("{}{}", &name, &append);
    let mut filename = directory.clone().join(filename_fmt);
    filename.set_extension("json");
    let mut file = File::create(&filename)?;

    let mut serde_shapes: Vec<SerdeShape> = Vec::new();

    for shape in shapes {
        let as_serde_points: Vec<SerdePoint3> = shape
            .0
            .iter()
            .map(|s| SerdePoint3 {
                x: s.x,
                y: s.y,
                z: s.z,
            })
            .collect();
        let serde_shape = SerdeShape {
            points: as_serde_points,
            tris: shape.1,
        };
        serde_shapes.push(serde_shape);
    }

    if args.log_success {
        println!("Writing file {:?}", &filename);
    }

    let json = serde_json::to_string_pretty(&ShapeCollection {
        shapes: serde_shapes,
    })
    .expect("Failed to serialize shape data.");
    file.write_all(json.as_bytes())
        .expect("Failed to write json to disk.");

    Ok(())
}

fn write_mesh_to_obj(
    directory: &Path,
    name: &str,
    append: &str,
    shape: (Vec<Point3<f32>>, Vec<[u32; 3]>),
    args: &Args,
) -> std::io::Result<()> {
    let mut file_cont: Vec<String> = Vec::new();
    let name_fmt = format!("o {}", &name);
    file_cont.push(name_fmt);

    for v in shape.0 {
        let fmt = format!("v {} {} {}", v.x, v.y, v.z);
        file_cont.push(fmt);
    }

    for tri in shape.1 {
        let fmt = format!("f {} {} {}", tri[0] + 1, tri[1] + 1, tri[2] + 1);
        file_cont.push(fmt);
    }

    let filename_fmt = format!("{}{}", &name, &append);
    let mut filename = directory.clone().join(filename_fmt);
    filename.set_extension("obj");

    let mut file = File::create(&filename)?;

    for line in file_cont {
        file.write(line.as_bytes())
            .expect("Failed to write to file.");
        file.write("\n".as_bytes())
            .expect("Failed to write to file.");
    }

    if args.log_success {
        println!("Writing file {:?}", &filename);
    }

    Ok(())
}
