import os
import subprocess
import argparse

if __name__ == '__main__':

    ##parser = argparse.ArgumentParser(description='Run COLMAP for multi-view stereo')
    ##parser.add_argument('--data_dir', type=str, required=True, help='Path to the input directory containing images in data_dir/images')
    ##args = parser.parse_args()
    data_dir = 'data/chair'

    # Feature extraction with shared intrinsics (assume it's the same camera)
    subprocess.run(['D:\\CG\\colmap\\bin\\colmap.exe', 'feature_extractor', '--image_path', os.path.join(data_dir, 'images'), '--database_path', os.path.join(data_dir, 'database.db'), '--ImageReader.single_camera', '1', '--ImageReader.camera_model', 'PINHOLE', '--SiftExtraction.use_gpu', '1'])

    # Feature matching
    subprocess.run(['D:\\CG\\colmap\\bin\\colmap.exe', 'exhaustive_matcher', '--database_path', os.path.join(data_dir, 'database.db'), '--SiftMatching.use_gpu', '1'])

    # Create sparse reconstruction folder
    os.makedirs(os.path.join(data_dir, 'sparse'), exist_ok=True)

    # Sparse reconstruction
    subprocess.run(['D:\\CG\\colmap\\bin\\colmap.exe', 'mapper', '--image_path', os.path.join(data_dir, 'images'), '--database_path', os.path.join(data_dir, 'database.db'), '--output_path', os.path.join(data_dir, 'sparse')])

    # Convert binary model to text format
    os.makedirs(os.path.join(data_dir, 'sparse', '0_text'), exist_ok=True)
    subprocess.run(['D:\\CG\\colmap\\bin\\colmap.exe', 'model_converter', '--input_path', os.path.join(data_dir, 'sparse', '0'), '--output_path', os.path.join(data_dir, 'sparse', '0_text'), '--output_type', 'TXT'])

    print("COLMAP multi-view stereo pipeline completed successfully!")
    print("Sparse 3D reconstruction saved in:", os.path.join(data_dir, 'sparse', '0_text'))
    