import os
import time
import shutil
import cv2
import multiprocessing
from pathlib import Path
import m3u8
import numpy as np

# --- Configuration Constants ---
INPUT_BASE_DIR = "tmp_hls"
OUTPUT_BASE_DIR = "tmp_hls_proc"
SEGMENT_EXTENSION = ".ts"
PLAYLIST_FILENAME = "playlist.m3u8"
PROCESSED_PLAYLIST_FILENAME = "playlist.m3u8"
PROCESSED_SEGMENT_SUFFIX = "_detected"  # Changed from _blurred
POLL_INTERVAL_STREAM_PROCESS = 1
POLL_INTERVAL_MAIN_MONITOR = 5
VIDEO_FOURCC = 'X264'  # FourCC code for encoding. 'X264' is for x264. Other options: 'H264', 'avc1'.
PROCESSED_PLAYLIST_BASE_URI = "http://127.0.0.1:8000/processed_hls"

MOBILENET_PROTOTXT_PATH = "model/MobileNetSSD_deploy.prototxt.txt"
MOBILENET_MODEL_PATH = "model/MobileNetSSD_deploy.caffemodel"

MOBILENET_CLASS_NAMES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor"
]

CONFIDENCE_THRESHOLD = 0.3 
INPUT_SIZE_MOBILENET = (300, 300)
MEAN_SUBTRACTION_MOBILENET = (127.5, 127.5, 127.5)

LOADED_CLASS_NAMES = None

def load_mobilenetssd_model_and_names():
    """
    Loads the MobileNet SSD model and class names.
    Returns the network and class names, or None, None if loading fails.
    """
    global LOADED_CLASS_NAMES
    pid = os.getpid()
    net = None

    try:
        print(f"[{pid}] Loading MobileNet SSD model files: Prototxt='{MOBILENET_PROTOTXT_PATH}', Model='{MOBILENET_MODEL_PATH}'")
        if not os.path.exists(MOBILENET_PROTOTXT_PATH):
            print(f"[{pid}] ERROR: MobileNet SSD prototxt file not found: {MOBILENET_PROTOTXT_PATH}")
            return None, None
        if not os.path.exists(MOBILENET_MODEL_PATH):
            print(f"[{pid}] ERROR: MobileNet SSD model file not found: {MOBILENET_MODEL_PATH}")
            return None, None
        
        net = cv2.dnn.readNetFromCaffe(MOBILENET_PROTOTXT_PATH, MOBILENET_MODEL_PATH)
        print(f"[{pid}] MobileNet SSD model loaded. Configuring backend/target...")


        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"[{pid}] MobileNetSSD: Successfully set backend and target to OpenCV/CPU.")

        LOADED_CLASS_NAMES = MOBILENET_CLASS_NAMES
        print(f"[{pid}] Loaded {len(LOADED_CLASS_NAMES)} class names for MobileNet SSD.")

        return net, LOADED_CLASS_NAMES

    except cv2.error as e_cv_load:
        print(f"[{pid}] Error loading MobileNet SSD model (OpenCV Error): {e_cv_load}")
        return None, None
    except Exception as e_load_main:
        print(f"[{pid}] General error during MobileNet SSD model loading: {e_load_main}")
        return None, None


def apply_mobilenetssd_to_segment(input_segment_path, output_segment_path, net, class_names_local):
    """
    Reads a video segment, applies MobileNet SSD object detection, and saves the result.
    """
    pid = os.getpid()
    if net is None or class_names_local is None:
        print(f"[{pid}] MobileNet SSD model or class names not loaded. Skipping detection for {input_segment_path}.")
        return False

    cap = cv2.VideoCapture(str(input_segment_path))
    if not cap.isOpened():
        print(f"[{pid}] Error: Could not open video segment: {input_segment_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        print(f"[{pid}] Warning: Invalid FPS {fps} for {input_segment_path}. Defaulting to 25 FPS.")
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
    out_writer = cv2.VideoWriter(str(output_segment_path), fourcc, fps, (frame_width, frame_height))

    if not out_writer.isOpened():
        print(f"[{pid}] Error: Could not open VideoWriter for {output_segment_path}.")
        cap.release()
        return False

    processed_frames_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create blob from frame for MobileNet SSD
        # Input image is resized to 300x300, and mean values are subtracted.
        # The 0.007843 is 1/127.5, which is a common scaling factor.
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, INPUT_SIZE_MOBILENET),  # Resize to 300x300
            0.007843,                                # Scale factor (1/127.5)
            INPUT_SIZE_MOBILENET,                    # Spatial size
            MEAN_SUBTRACTION_MOBILENET               # Mean subtraction values
        )
        net.setInput(blob)
        detections = net.forward() # Detections are [1, 1, N, 7] where N is number of detections

        # Loop over the detections
        for i in range(detections.shape[2]): # N detections
            confidence = detections[0, 0, i, 2] # Confidence is at index 2

            if confidence > CONFIDENCE_THRESHOLD:
                class_id = int(detections[0, 0, i, 1]) # Class ID is at index 1
                
                # Box coordinates are normalized, so multiply by frame dimensions
                box_x = int(detections[0, 0, i, 3] * frame_width)
                box_y = int(detections[0, 0, i, 4] * frame_height)
                box_width = int(detections[0, 0, i, 5] * frame_width)
                box_height = int(detections[0, 0, i, 6] * frame_height)

                # Draw bounding box and label
                color = (0, 255, 0) # Green
                try:
                    label = f"{class_names_local[class_id]}: {confidence:.2f}"
                except IndexError:
                    label = f"ClassID {class_id}: {confidence:.2f}" # Fallback if class_id is out of bounds
                
                cv2.rectangle(frame, (box_x, box_y), (box_width, box_height), color, 2)
                cv2.putText(frame, label, (box_x, box_y - 10 if box_y -10 > 10 else box_y + 10) , 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        out_writer.write(frame)
        processed_frames_count += 1

    cap.release()
    out_writer.release()

    if processed_frames_count > 0:
        return True
    else:
        print(f"[{pid}] Warning: No frames processed for {input_segment_path}.")
        if os.path.exists(output_segment_path):
            try: os.remove(output_segment_path)
            except OSError as e: print(f"[{pid}] Error deleting empty output {output_segment_path}: {e}")
        return False


def update_processed_playlist(stream_input_dir, stream_output_dir, suffix, playlist_base_uri_config):
    """
    Loads the original playlist, modifies segment URIs to point to detected versions,
    and saves the new playlist.
    """
    original_playlist_path = stream_input_dir / PLAYLIST_FILENAME
    processed_playlist_path = stream_output_dir / PROCESSED_PLAYLIST_FILENAME
    stream_folder_name = stream_output_dir.name
    pid = os.getpid()

    if not original_playlist_path.exists():
        return

    try:
        playlist = m3u8.load(str(original_playlist_path))

        for segment in playlist.segments:
            original_uri = segment.uri
            original_filename = os.path.basename(original_uri)
            
            base_name, ext = os.path.splitext(original_filename)
            if not base_name.endswith(suffix):
                 processed_segment_filename = base_name + suffix + ext
            else:
                 processed_segment_filename = original_filename


            if playlist_base_uri_config:
                prefix = playlist_base_uri_config
                if not prefix.endswith('/'):
                    prefix += '/'
                segment.uri = f"{prefix}{stream_folder_name}/{processed_segment_filename}"
            else:
                segment.uri = processed_segment_filename
        
        processed_playlist_path.parent.mkdir(parents=True, exist_ok=True)
        playlist.dump(str(processed_playlist_path))

    except Exception as e:
        print(f"[{pid}] Error updating playlist using m3u8 library for {stream_folder_name}: {e}")


def process_stream_folder(stream_input_path_str, stream_output_path_str, stop_event):
    stream_input_dir = Path(stream_input_path_str)
    stream_output_dir = Path(stream_output_path_str)
    pid = os.getpid()
    print(f"[{pid}] Starting to process stream: {stream_input_dir} with MobileNet SSD")

    # Load MobileNet SSD model for this process
    model_net, class_names_for_process = load_mobilenetssd_model_and_names()
    if model_net is None:
        print(f"[{pid}] CRITICAL: Failed to load MobileNet SSD model for stream {stream_input_dir}. This process will not detect objects.")
    
    stream_output_dir.mkdir(parents=True, exist_ok=True)
    processed_segment_basenames = set()

    try:
        while not stop_event.is_set():
            if not stream_input_dir.exists():
                print(f"[{pid}] Input stream folder {stream_input_dir} deleted. Exiting process.")
                break

            current_input_segment_files = {f.name for f in stream_input_dir.glob(f"*{SEGMENT_EXTENSION}") if f.is_file()}
            current_input_segment_basenames = {Path(f).stem for f in current_input_segment_files}
            
            segments_to_remove_from_output = processed_segment_basenames - current_input_segment_basenames
            for base_to_remove in list(segments_to_remove_from_output): 
                output_segment_name = base_to_remove + PROCESSED_SEGMENT_SUFFIX + SEGMENT_EXTENSION
                output_segment_path = stream_output_dir / output_segment_name
                if output_segment_path.exists():
                    try: output_segment_path.unlink()
                    except OSError as e: print(f"[{pid}] Error deleting stale output {output_segment_path}: {e}")
                processed_segment_basenames.discard(base_to_remove)

            new_segments_processed_this_cycle = False
            for segment_basename in current_input_segment_basenames:
                if segment_basename not in processed_segment_basenames:
                    input_segment_file = stream_input_dir / (segment_basename + SEGMENT_EXTENSION)
                    output_segment_file = stream_output_dir / (segment_basename + PROCESSED_SEGMENT_SUFFIX + SEGMENT_EXTENSION)
                    
                    if model_net: # Only process if model is loaded
                        if apply_mobilenetssd_to_segment(input_segment_file, output_segment_file, model_net, class_names_for_process):
                            processed_segment_basenames.add(segment_basename)
                            new_segments_processed_this_cycle = True
                        else:
                            print(f"[{pid}] Failed to process segment with MobileNet SSD: {input_segment_file}")
                            if output_segment_file.exists():
                                try: output_segment_file.unlink()
                                except OSError: pass 
                    else:
                        print(f"[{pid}] MobileNet SSD model not loaded, skipping processing for {input_segment_file}")

            processed_playlist_file = stream_output_dir / PROCESSED_PLAYLIST_FILENAME
            if new_segments_processed_this_cycle or not processed_playlist_file.exists() or segments_to_remove_from_output:
                update_processed_playlist(
                    stream_input_dir, stream_output_dir,
                    PROCESSED_SEGMENT_SUFFIX, PROCESSED_PLAYLIST_BASE_URI
                )
            time.sleep(POLL_INTERVAL_STREAM_PROCESS)
    except Exception as e:
        print(f"[{pid}] Unexpected error in process_stream_folder for {stream_input_dir}: {type(e).__name__} - {e}")
    finally:
        print(f"[{pid}] Process for {stream_input_dir} stopping.")
        if 'model_net' in locals() and model_net is not None:
            del model_net # Hint for garbage collection


def main_monitor():
    input_base = Path(INPUT_BASE_DIR)
    output_base = Path(OUTPUT_BASE_DIR)
    input_base.mkdir(parents=True, exist_ok=True)
    output_base.mkdir(parents=True, exist_ok=True)

    print(f"Main monitor started. Watching {input_base} for stream folders.")
    print(f"Processed streams will be saved in {output_base}.")
    print(f"Using MobileNet SSD: Prototxt='{MOBILENET_PROTOTXT_PATH}', Model='{MOBILENET_MODEL_PATH}'")
    print(f"IMPORTANT: Ensure these MobileNet SSD model files exist and are accessible.")
    if PROCESSED_PLAYLIST_BASE_URI:
        print(f"Processed playlists will use base URI: {PROCESSED_PLAYLIST_BASE_URI}")
    else:
        print(f"Processed playlists will use relative segment URIs.")
    print("Make sure 'm3u8' and 'numpy' Python libraries are installed (pip install m3u8 numpy opencv-python).")
    print(f"Press Ctrl+C to stop.")

    print("[Main] Performing startup cleanup of output directory...")
    if output_base.exists():
        current_input_stream_names = {d.name for d in input_base.iterdir() if d.is_dir()}
        for output_stream_dir in output_base.iterdir():
            if output_stream_dir.is_dir() and output_stream_dir.name not in current_input_stream_names:
                print(f"[Main] Startup: Removing output for non-existent input stream '{output_stream_dir.name}'")
                try: shutil.rmtree(output_stream_dir)
                except OSError as e: print(f"[Main] Startup: Error removing {output_stream_dir}: {e}")
    print("[Main] Startup cleanup complete.")
    
    active_processes = {} 

    try:
        while True:
            current_stream_folders_on_disk = {d.name for d in input_base.iterdir() if d.is_dir()}
            
            for stream_name in current_stream_folders_on_disk:
                if stream_name not in active_processes:
                    print(f"[Main] New stream folder detected: {stream_name}")
                    stream_input_path = input_base / stream_name
                    stream_output_path = output_base / stream_name
                    stop_event = multiprocessing.Event()
                    process = multiprocessing.Process(
                        target=process_stream_folder,
                        args=(str(stream_input_path), str(stream_output_path), stop_event)
                    )
                    process.start()
                    active_processes[stream_name] = (process, stop_event)
                    print(f"[Main] Started process {process.pid} for stream {stream_name}")

            active_stream_names_in_memory = list(active_processes.keys())
            for stream_name in active_stream_names_in_memory:
                if stream_name not in current_stream_folders_on_disk:
                    print(f"[Main] Stream folder '{stream_name}' deleted. Stopping its process.")
                    process, stop_event = active_processes.pop(stream_name)
                    stop_event.set() 
                    process.join(timeout=10) 
                    if process.is_alive():
                        print(f"[Main] Process for {stream_name} (PID {process.pid}) unresponsive. Terminating.")
                        process.terminate() 
                        process.join() 
                    print(f"[Main] Process for {stream_name} stopped.")
                    output_dir_to_remove = output_base / stream_name
                    if output_dir_to_remove.exists():
                        print(f"[Main] Removing output directory: {output_dir_to_remove}")
                        try: shutil.rmtree(output_dir_to_remove)
                        except OSError as e: print(f"[Main] Error removing {output_dir_to_remove}: {e}")
            time.sleep(POLL_INTERVAL_MAIN_MONITOR)
    except KeyboardInterrupt:
        print("\n[Main] KeyboardInterrupt. Shutting down...")
    except Exception as e:
        print(f"[Main] Unexpected error in main_monitor: {type(e).__name__} - {e}")
    finally:
        print("[Main] Shutting down all stream processes...")
        for stream_name, (process, stop_event) in active_processes.items():
            print(f"[Main] Stopping process for {stream_name} (PID {process.pid})...")
            stop_event.set()
            process.join(timeout=5) 
            if process.is_alive():
                print(f"[Main] Force terminating process for {stream_name} (PID {process.pid})...")
                process.terminate()
                process.join() 
        print("[Main] All stream processes shut down. Exiting.")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    main_monitor()