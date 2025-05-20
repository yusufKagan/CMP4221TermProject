import httpx
import subprocess
import threading


class StreamObject:
    def __init__(self,video_path, api_url="http://localhost:8000", port=5000, host="127.0.0.1"):
        self.api_url = api_url
        self.stored_ssrc = None
        self.gstreamer_thread = None
        self.gstreamer_process = None
        self.port = port
        self.host = host
        self.video_path = video_path

    def __str__(self):
        return f"StreamObject(SSRC={self.stored_ssrc})"


    def request_ssrc(self):
        try:
            response = httpx.post(f"{self.api_url}/streams/request_ssrc")
            response.raise_for_status()
            data = response.json()
            self.stored_ssrc = data["ssrc"]
            print(f"[INFO] SSRC acquired and stored: {self.stored_ssrc}")
        except Exception as e:
            print(f"[ERROR] Failed to request SSRC: {e}")

    def start_stream(self):
        if not self.stored_ssrc:
            print("[ERROR] SSRC not available. Request SSRC first.")
            return
        try:
            response = httpx.post(f"{self.api_url}/streams/start", params={"ssrc": self.stored_ssrc})
            print(response.json())
        except Exception as e:
            print(f"[ERROR] Failed to start stream: {e}")

    def stop_stream(self):
        if not self.stored_ssrc:
            print("[ERROR] SSRC not available. Request SSRC first.")
            return
        try:
            if self.gstreamer_process and self.gstreamer_process.poll() is None:
                print("[INFO] Terminating GStreamer process...")
                self.gstreamer_process.terminate()
                print("[INFO] GStreamer process terminated.")
                self.gstreamer_process = None
                if self.gstreamer_thread and self.gstreamer_thread.is_alive():
                    self.gstreamer_thread.join(timeout=1)
                    if self.gstreamer_thread.is_alive():
                        print("[WARN] GStreamer thread did not join gracefully.")
                    self.gstreamer_thread = None

            
            response = httpx.post(f"{self.api_url}/streams/stop", params={"ssrc": self.stored_ssrc})
            response.raise_for_status()
            print("[INFO] Stream stop request sent.")
            print("response json", response.json())
            
            self.request_ssrc()
            self.start_gstreamer_thread()



        except httpx.RequestError as e:
            print(f"[ERROR] Failed to connect to the API: {e}")
        except httpx.HTTPStatusError as e:
            print(f"[ERROR] Failed to stop stream: {e}")
            try:
                print(f"[ERROR] Response: {e.response.json()}")
            except:
                print(f"[ERROR] Response: {e.response.text}")
        except Exception as e:
            print(f"[ERROR] An unexpected error occurred while stopping the stream: {e}")


    def adjust_stream(self):
        if not self.stored_ssrc:
            print("[ERROR] SSRC not available. Request SSRC first.")
            return

        bitrate = input("Bitrate (kbps) [leave blank to skip]: ")
        width = input("Resolution Width [leave blank to skip]: ")
        height = input("Resolution Height [leave blank to skip]: ")
        fr_n = input("Framerate Numerator [leave blank to skip]: ")
        fr_d = input("Framerate Denominator [leave blank to skip]: ")

        params = {"ssrc": self.stored_ssrc}
        if bitrate:
            params["bitrate_kbps"] = int(bitrate)
        if width:
            params["resolution_width"] = int(width)
        if height:
            params["resolution_height"] = int(height)
        if fr_n:
            params["framerate_numerator"] = int(fr_n)
        if fr_d:
            params["framerate_denominator"] = int(fr_d)

        try:
            response = httpx.post(f"{self.api_url}/streams/adjust", params=params)
            print(response.json())
        except Exception as e:
            print(f"[ERROR] Failed to adjust stream: {e}")

    def get_status(self):
        try:
            if not self.stored_ssrc:
                print("[INFO] No SSRC provided, fetching system-wide status...")
                response = httpx.get(f"{self.api_url}/streams/status")
            else:
                response = httpx.get(f"{self.api_url}/streams/status", params={"ssrc": self.stored_ssrc})
            print(response.json())
        except Exception as e:
            print(f"[ERROR] Failed to get status: {e}")

    def launch_gstreamer_pipeline(self):
        print(f"[INFO] Launching GStreamer pipeline with SSRC={self.stored_ssrc} to {self.host}:{self.port}...")
        if not self.video_path:
            print("no video path")
            return
        gst_command = [
            "gst-launch-1.0",
            "filesrc", f"location={self.video_path}",
            "!", "decodebin",
            "!", "videoconvert",
            "!", "x264enc", "tune=zerolatency", "bitrate=300", "speed-preset=fast",
            "!", "rtph264pay", "config-interval=1",
            "!", f"application/x-rtp,ssrc=(uint){self.stored_ssrc}",
            "!", "udpsink", f"host={self.host}", f"port={self.port}", "auto-multicast=false", "ts-offset=0"
        ]
        
        try:
            self.gstreamer_process = subprocess.Popen(
                gst_command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
                )
            self.gstreamer_process.wait()
        except Exception as e:
            print(f"[ERROR] GStreamer pipeline failed: {e}")

    def start_gstreamer_thread(self):
        if not self.stored_ssrc:
            print("[ERROR] Cannot start GStreamer without SSRC.")
            return

        if self.gstreamer_thread is None or not self.gstreamer_thread.is_alive():
            self.gstreamer_thread = threading.Thread(target=self.launch_gstreamer_pipeline)
            self.gstreamer_thread.daemon = True
            self.gstreamer_thread.start()
            print("[INFO] GStreamer pipeline started in background.")
        else:
            print("[INFO] GStreamer pipeline is already running.")

    def cleanup(self):
        if self.gstreamer_process and self.gstreamer_process.poll() is None:
            print("[INFO] Terminating GStreamer pipeline...")
            self.gstreamer_process.terminate()
            try:
                self.gstreamer_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("[WARN] GStreamer did not terminate gracefully, killing...")
                self.gstreamer_process.kill()


stream_object_list =[]

def create_stream_object():
    global stream_object_list
    video_path = input("Enter the path to the video file: ").strip()
    if not video_path:
        print("[ERROR] Video path cannot be empty.")
        return
    stream = StreamObject(video_path=video_path)
    stream_object_list.append(stream)
    stream.request_ssrc()
    stream.start_gstreamer_thread()

def delete_stream_object(index):
    global stream_object_list
    try:
        stream = stream_object_list.pop(index)
        stream.cleanup()
        print(f"[INFO] Deleted and cleaned up stream at index {index}")
    except IndexError:
        print("[ERROR] Invalid index.")
        
def list_streams():
    global stream_object_list
    if not stream_object_list:
        print("[INFO] No active streams.")
        return False
    for idx, stream in enumerate(stream_object_list):
        print(f"{idx}: {stream}")
    
    return True

def stream_menu(stream):
    global stream_object_list
    options = {
        "1": ("Start Stream", stream.start_stream),
        "2": ("Stop Stream", stream.stop_stream),
        "3": ("Adjust Stream Parameters", stream.adjust_stream),
        "4": ("Get Stream/System Status", stream.get_status),
        "b": ("Back to Stream Selection", None)
    }

    while True:
        print(f"\n--- Control Menu for {stream} ---")
        for key, (desc, _) in options.items():
            print(f"{key}: {desc}")
        choice = input("Choose an action: ").strip()
        if choice == "b":
            break
        elif choice in options:
            try:
                options[choice][1]()
            except Exception as e:
                print(f"[ERROR] {e}")
        else:
            print("Invalid option.")

def main_menu():
    global stream_object_list
    options = {
        "1": "Create New Stream",
        "2": "Select Stream to Control",
        "3": "Delete Stream",
        "4": "List All Streams",
        "q": "Quit"
    }

    while True:
        print("\n--- Stream Manager Menu ---")
        for key, desc in options.items():
            print(f"{key}: {desc}")
        choice = input("Select an option: ").strip()

        if choice == "1":
            create_stream_object()
        elif choice == "2":
            if list_streams():
                try:
                    idx = int(input("Enter stream index: "))
                    stream_menu(stream_object_list[idx])
                except (ValueError, IndexError):
                    print("[ERROR] Invalid index.")
        elif choice == "3":
            list_streams()
            try:
                idx = int(input("Enter stream index to delete: "))
                delete_stream_object(idx)
            except ValueError:
                print("[ERROR] Invalid input.")
        elif choice == "4":
            list_streams()
        elif choice == "q":
            print("[INFO] Cleaning up all streams before exit...")
            for stream in stream_object_list:
                stream.cleanup()
            break
        else:
            print("Invalid option.")

if __name__ == "__main__":
    main_menu()
