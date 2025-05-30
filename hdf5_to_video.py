import h5py
import numpy as np
import cv2
import os
import argparse
from tqdm import tqdm

def hdf5_to_video_multi_camera(hdf5_path, output_dir, dataset_names=None, fps=30, codec='mp4v', layout='separate', camera_info_count=3):
    """
    Convert HDF5 multi-camera data to video(s)
    
    Args:
    hdf5_path: Path to HDF5 file
    output_dir: Output directory for video files
    dataset_names: List of dataset names for each camera (auto-detect if None)
    fps: Video frame rate
    codec: Video codec
    layout: 'separate', 'side_by_side', 'top_bottom', or 'camera_grouped' (group info by camera)
    camera_info_count: Number of info streams per camera (default: 3)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(hdf5_path, 'r') as f:
        print(f"HDF5 file contents: {list(f.keys())}")
        
        # Auto-detect camera datasets if not specified
        if dataset_names is None:
            dataset_names = []
            for key in f.keys():
                data = f[key]
                if isinstance(data, h5py.Dataset) and len(data.shape) >= 3:
                    dataset_names.append(key)
            
            if len(dataset_names) == 0:
                raise ValueError("Cannot find suitable image datasets.")
            
            print(f"Auto-detected datasets: {dataset_names}")
        
        # Load all camera data
        camera_data = {}
        for dataset_name in dataset_names:
            if dataset_name not in f:
                raise ValueError(f"Dataset '{dataset_name}' not found in HDF5 file")
            
            data = f[dataset_name]
            camera_data[dataset_name] = data
            print(f"Camera '{dataset_name}' - Shape: {data.shape}, Type: {data.dtype}")
        
        # Validate data consistency
        shapes = [data.shape for data in camera_data.values()]
        if not all(shape[0] == shapes[0][0] for shape in shapes):
            raise ValueError("All cameras must have the same number of frames")
        
        num_frames = shapes[0][0]
        print(f"Total frames: {num_frames}")
        
        if layout == 'separate':
            # Create separate video for each camera
            _create_separate_videos(camera_data, output_dir, fps, codec)
        elif layout == 'side_by_side':
            # Create combined side-by-side video
            _create_combined_video(camera_data, output_dir, fps, codec, 'horizontal')
        elif layout == 'top_bottom':
            # Create combined top-bottom video
            _create_combined_video(camera_data, output_dir, fps, codec, 'vertical')
        elif layout == 'camera_grouped':
            # Create camera-grouped layout (2x3 grid)
            _create_camera_grouped_video(camera_data, output_dir, fps, codec, camera_info_count)
        else:
            raise ValueError("Layout must be 'separate', 'side_by_side', 'top_bottom', or 'camera_grouped'")

def _create_separate_videos(camera_data, output_dir, fps, codec):
    """Create separate video for each camera"""
    for camera_name, data in camera_data.items():
        output_path = os.path.join(output_dir, f"{camera_name}.mp4")
        print(f"\nProcessing camera: {camera_name}")
        
        # Get video parameters
        if len(data.shape) == 3:  # Grayscale
            num_frames, height, width = data.shape
            is_color = False
        elif len(data.shape) == 4:  # Color
            num_frames, height, width, channels = data.shape
            is_color = True
        else:
            raise ValueError(f"Unsupported data shape for {camera_name}: {data.shape}")
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), is_color)
        
        if not out.isOpened():
            raise RuntimeError(f"Cannot create video file: {output_path}")
        
        # Process frames
        for i in tqdm(range(num_frames), desc=f"Processing {camera_name}"):
            frame = _process_frame(data[i], is_color)
            out.write(frame)
        
        out.release()
        print(f"Video saved: {output_path}")

def _create_camera_grouped_video(camera_data, output_dir, fps, codec, camera_info_count):
    """Create camera-grouped video layout (e.g., 2x3 grid for 2 cameras with 3 info each)"""
    dataset_names = list(camera_data.keys())
    
    # Group datasets by camera (extract camera name from suffix)
    cameras = {}
    for dataset_name in dataset_names:
        # Extract camera name from suffix (e.g., "color_gripper" -> "gripper")
        if '_' in dataset_name:
            parts = dataset_name.split('_')
            # Take the last part as camera name
            camera_name = parts[-1]
            # If there are multiple underscores, might need different logic
            if camera_name == 'colorized':  # Handle special case like "depth_gripper_colorized"
                camera_name = parts[-2]  # Use "gripper" instead
        else:
            # If no underscore, use the whole name as camera
            camera_name = dataset_name
        
        if camera_name not in cameras:
            cameras[camera_name] = []
        cameras[camera_name].append(dataset_name)
    
    # Sort datasets within each camera group for consistent ordering
    for camera_name in cameras:
        cameras[camera_name].sort()  # This ensures consistent ordering
    
    print(f"Detected camera groups: {dict(cameras)}")
    
    # Sort cameras for consistent ordering
    camera_names = sorted(cameras.keys())
    
    # Validate that each camera has the expected number of info streams
    for camera_name in camera_names:
        if len(cameras[camera_name]) != camera_info_count:
            print(f"Warning: Camera {camera_name} has {len(cameras[camera_name])} streams, expected {camera_info_count}")
    
    output_path = os.path.join(output_dir, "camera_grouped.mp4")
    print(f"\nCreating camera-grouped video layout")
    
    # Print the actual layout that will be created
    print("Layout will be:")
    for camera_name in camera_names:
        row_info = " | ".join(cameras[camera_name][:camera_info_count])
        print(f"  {row_info}")
    
    # Get dimensions from first dataset
    first_dataset = cameras[camera_names[0]][0]
    first_data = camera_data[first_dataset]
    
    if len(first_data.shape) == 3:
        num_frames, height, width = first_data.shape
    else:
        num_frames, height, width, channels = first_data.shape
    
    # Calculate combined frame size (info_count horizontally, cameras vertically)
    combined_width = width * camera_info_count
    combined_height = height * len(camera_names)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height), True)
    
    if not out.isOpened():
        raise RuntimeError(f"Cannot create video file: {output_path}")
    
    # Process frames
    for i in tqdm(range(num_frames), desc="Processing camera-grouped video"):
        camera_rows = []
        
        for camera_name in camera_names:
            # Arrange all info for this camera horizontally in sorted order
            camera_frames = []
            sorted_datasets = sorted(cameras[camera_name])[:camera_info_count]
            
            for dataset_name in sorted_datasets:
                frame = _process_frame(camera_data[dataset_name][i], True)
                # Resize if necessary to match expected dimensions
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                camera_frames.append(frame)
            
            # If fewer streams than expected, pad with black frames
            while len(camera_frames) < camera_info_count:
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                camera_frames.append(black_frame)
            
            # Stack horizontally for this camera
            camera_row = np.hstack(camera_frames)
            camera_rows.append(camera_row)
        
        # Combine camera rows vertically
        combined_frame = np.vstack(camera_rows)
        out.write(combined_frame)
    
    out.release()
    print(f"Camera-grouped video saved: {output_path}")

def _create_combined_video(camera_data, output_dir, fps, codec, direction):
    """Create combined video with multiple camera views"""
    camera_names = list(camera_data.keys())
    
    if len(camera_names) != 2:
        print(f"Warning: Combined layout works best with 2 cameras, found {len(camera_names)}")
    
    output_path = os.path.join(output_dir, f"combined_{direction}.mp4")
    print(f"\nCreating combined video: {direction}")
    
    # Get dimensions from first camera
    first_data = list(camera_data.values())[0]
    if len(first_data.shape) == 3:
        num_frames, height, width = first_data.shape
        is_color = False
    else:
        num_frames, height, width, channels = first_data.shape
        is_color = True
    
    # Calculate combined frame size
    if direction == 'horizontal':
        combined_width = width * len(camera_names)
        combined_height = height
    else:  # vertical
        combined_width = width
        combined_height = height * len(camera_names)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (combined_width, combined_height), True)
    
    if not out.isOpened():
        raise RuntimeError(f"Cannot create video file: {output_path}")
    
    # Process frames
    for i in tqdm(range(num_frames), desc="Processing combined video"):
        frames = []
        for camera_name in camera_names:
            frame = _process_frame(camera_data[camera_name][i], True)  # Force color for combination
            frames.append(frame)
        
        # Combine frames
        if direction == 'horizontal':
            combined_frame = np.hstack(frames)
        else:  # vertical
            combined_frame = np.vstack(frames)
        
        out.write(combined_frame)
    
    out.release()
    print(f"Combined video saved: {output_path}")

def _process_frame(frame, is_color):
    """Process individual frame"""
    # Data type conversion
    if frame.dtype != np.uint8:
        frame_min, frame_max = frame.min(), frame.max()
        if frame_max > frame_min:
            frame = ((frame - frame_min) / (frame_max - frame_min) * 255).astype(np.uint8)
        else:
            frame = np.zeros_like(frame, dtype=np.uint8)
    
    # Handle color channels
    if is_color:
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # RGB to BGR for OpenCV
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame.squeeze(), cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        if len(frame.shape) == 3:
            frame = frame.squeeze()
        # For separate videos, keep grayscale; for combined, convert to BGR
        if is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    
    return frame

def hdf5_to_video(hdf5_path, output_path, dataset_name=None, fps=30, codec='mp4v'):
    """
    Single camera version (backward compatibility)
    """
    output_dir = os.path.dirname(output_path) or '.'
    if dataset_name:
        dataset_names = [dataset_name]
    else:
        dataset_names = None
    
    hdf5_to_video_multi_camera(hdf5_path, output_dir, dataset_names, fps, codec, 'separate')

def list_hdf5_contents(hdf5_path):
    """List HDF5 file structure"""
    def print_structure(name, obj):
        indent = "  " * name.count('/')
        if isinstance(obj, h5py.Dataset):
            print(f"{indent}{name}: Dataset {obj.shape} {obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"{indent}{name}: Group")
    
    with h5py.File(hdf5_path, 'r') as f:
        print(f"HDF5 file structure: {hdf5_path}")
        print("-" * 50)
        f.visititems(print_structure)

def main():
    parser = argparse.ArgumentParser(description='Convert HDF5 multi-camera data to video')
    parser.add_argument('input', help='Input HDF5 file path')
    parser.add_argument('output', help='Output directory for video files')
    parser.add_argument('--datasets', '-d', nargs='+', help='Dataset names for cameras (e.g., camera1 camera2)')
    parser.add_argument('--fps', type=int, default=30, help='Video frame rate (default: 30)')
    parser.add_argument('--codec', default='mp4v', help='Video codec (default: mp4v)')
    parser.add_argument('--layout', choices=['separate', 'side_by_side', 'top_bottom', 'camera_grouped'], 
                       default='separate', help='Video layout (default: separate)')
    parser.add_argument('--camera-info-count', type=int, default=3, 
                       help='Number of info streams per camera for camera_grouped layout (default: 3)')
    parser.add_argument('--list', '-l', action='store_true', help='List HDF5 file contents')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file does not exist: {args.input}")
        return
    
    if args.list:
        list_hdf5_contents(args.input)
        return
    
    try:
        hdf5_to_video_multi_camera(
            hdf5_path=args.input,
            output_dir=args.output,
            dataset_names=args.datasets,
            fps=args.fps,
            codec=args.codec,
            layout=args.layout,
            camera_info_count=args.camera_info_count
        )
    except Exception as e:
        print(f"Conversion failed: {str(e)}")

# Usage examples
if __name__ == "__main__":
    main()
    
    # Example usage:
    # hdf5_to_video_multi_camera('data.h5', 'output/', layout='camera_grouped', camera_info_count=3)
    # hdf5_to_video_multi_camera('data.h5', 'output/', ['camera1_rgb', 'camera1_depth', 'camera1_seg', 'camera2_rgb', 'camera2_depth', 'camera2_seg'], layout='camera_grouped')