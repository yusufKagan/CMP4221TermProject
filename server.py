import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject # Added GObject for x264 tune parsing
import sys
import threading
import argparse
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import logging
import random
import shutil
from pathlib import Path
from typing import Dict, Any, Set, Optional, Tuple, List
import configparser # Added for config file handling

# --- Global Configuration & State ---
Gst.init(None)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Configuration Loading ---
config = configparser.ConfigParser()
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.ini"

# Default values (used if config file is missing or incomplete)
DEFAULT_SETTINGS = {
    "FastAPI": {
        "host": "127.0.0.1",
        "port": "8000",
    },
    "GStreamer": {
        "rtp_port": "5000",
        "hls_output_dir": "./tmp",
        "hls_processed_output_dir": "./tmp_proc",
        "default_bitrate_kbps": "1000",
        "default_x264_speed_preset_name": "ultrafast",
        "default_x264_tune_names": "zerolatency",
        "default_resolution_width": "640",
        "default_resolution_height": "480",
        "default_framerate_numerator": "25",
        "default_framerate_denominator": "1",
    },
    "HLS": {
        "target_duration": "2",
        "playlist_length": "10",
    }
}

def get_config_value(section: str, key: str, type_func=str):
    try:
        return type_func(config.get(section, key))
    except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
        logger.warning(f"Config value {section}/{key} not found or invalid, using default: {DEFAULT_SETTINGS[section][key]}")
        return type_func(DEFAULT_SETTINGS[section][key])

# --- Stream State Enum ---
class StreamState:
    REQUESTED = "REQUESTED"
    RECEIVING_MEDIA = "RECEIVING_MEDIA"
    STREAMING_HLS = "STREAMING_HLS"
    ERROR = "ERROR"

# --- GStreamer x264enc Parameter Mapping ---
X264_SPEED_PRESETS = {
    "ultrafast": 1, "superfast": 2, "veryfast": 3, "faster": 4,
    "fast": 5, "medium": 6, "slow": 7, "slower": 8, "veryslow": 9, "placebo": 10
}

# Fallback map for x264enc tune flags. Introspection is preferred.
# Values might differ based on GStreamer/x264enc version.
# Common values: film (1), animation (2), grain (4), stillimage (8),
# fastdecode (16 / 0x10), zerolatency (32 / 0x20).
# The previous map had zerolatency as 16, which might be incorrect for some versions.
X264_TUNE_FLAGS_FALLBACK = {
    "psnr": 0x0,
    "ssim": 0x40, # Example, check your gst-inspect-1.0 x264enc
    "film": 0x1,
    "animation": 0x2,
    "grain": 0x4, # Corrected from 0x3 if it was a typo
    "stillimage": 0x8,
    "fastdecode": 0x10, # 16
    "zerolatency": 0x20  # 32 - More commonly 0x20 for zerolatency
}

def parse_x264_tune_names(tune_names_str: str) -> int:
    """Parses a comma-separated string of tune names into the GStreamer x264enc tune flag value."""
    tune_value = 0
    if not tune_names_str or tune_names_str.lower() == "none":
        return 0

    try:
        x264_tune_gtype = GObject.GType.from_name("GstX264EncTune")
        if x264_tune_gtype:
            logger.info("Successfully got GType for GstX264EncTune.")
            enum_values = GObject.enum_type_get_values(x264_tune_gtype)
            name_to_val = {v.value_nick.lower(): v.value for v in enum_values} # Use lower for case-insensitivity
            logger.debug(f"Introspected GstX264EncTune values: {name_to_val}")

            parsed_successfully_via_introspection = False
            current_tune_value_introspection = 0
            for name in tune_names_str.split(','):
                name = name.strip().lower()
                if name in name_to_val:
                    current_tune_value_introspection |= name_to_val[name]
                    parsed_successfully_via_introspection = True
                else:
                    logger.warning(f"Unknown x264 tune name via introspection: '{name}'. Ignoring this part.")

            if parsed_successfully_via_introspection:
                logger.info(f"Using introspected x264 tune string '{tune_names_str}' to GFlags value: {current_tune_value_introspection}")
                return current_tune_value_introspection
            else:
                logger.warning(f"Introspection found GstX264EncTune, but tune names '{tune_names_str}' were not recognized in introspected values. Will attempt fallback.")
        else:
            logger.warning("Could not get GType for GstX264EncTune using GObject.GType.from_name. Falling back to predefined map.")
    except Exception as e:
        logger.warning(f"Error during GObject introspection for GstX264EncTune: {e}. Falling back to predefined map.")

    # Fallback logic
    logger.info(f"Using fallback map for x264 tune string '{tune_names_str}'.")
    fallback_tune_value = 0
    for name in tune_names_str.split(','):
        name = name.strip().lower()
        if name in X264_TUNE_FLAGS_FALLBACK:
            fallback_tune_value |= X264_TUNE_FLAGS_FALLBACK[name]
        else:
            logger.warning(f"Unknown x264 tune name in fallback map: '{name}'. Ignoring.")
    logger.info(f"Resulting GFlags value from fallback map: {fallback_tune_value}")
    return fallback_tune_value


# --- GStreamer Server Class ---
class GStreamerServer:
    def __init__(self, rtp_port: int, hls_output_dir: str, fastapi_host: str, fastapi_port: int, app_config: configparser.ConfigParser):
        self.rtp_port = rtp_port
        self.hls_output_dir = Path(hls_output_dir)
        self.fastapi_base_url = f"http://{fastapi_host}:{fastapi_port}"
        self.app_config = app_config 

        self.main_pipeline: Optional[Gst.Pipeline] = None
        self.rtpbin: Optional[Gst.Element] = None
        self.lock = threading.Lock()
        self.whitelisted_ssrcs: Set[str] = set()
        self.active_streams: Dict[str, Dict[str, Any]] = {} 
        self.gst_loop: Optional[GLib.MainLoop] = None
        self.gst_thread: Optional[threading.Thread] = None
        
        self.default_bitrate_kbps = get_config_value("GStreamer", "default_bitrate_kbps", int)
        speed_preset_name = get_config_value("GStreamer", "default_x264_speed_preset_name", str)
        self.default_x264_speed_preset = X264_SPEED_PRESETS.get(speed_preset_name.lower(), X264_SPEED_PRESETS["ultrafast"])
        
        tune_names_str = get_config_value("GStreamer", "default_x264_tune_names", str)
        self.default_x264_tune = parse_x264_tune_names(tune_names_str) # This will now log more details

        self.default_width = get_config_value("GStreamer", "default_resolution_width", int)
        self.default_height = get_config_value("GStreamer", "default_resolution_height", int)
        self.default_framerate_num = get_config_value("GStreamer", "default_framerate_numerator", int)
        self.default_framerate_den = get_config_value("GStreamer", "default_framerate_denominator", int)

        self.hls_target_duration = get_config_value("HLS", "target_duration", int)
        self.hls_playlist_length = get_config_value("HLS", "playlist_length", int)

        logger.info(f"GStreamerServer initialized. RTP Port: {rtp_port}, HLS Output: {self.hls_output_dir}")
        logger.info(f"Default Transcoding: {self.default_width}x{self.default_height}@{self.default_framerate_num}/{self.default_framerate_den}fps, {self.default_bitrate_kbps}kbps, preset={speed_preset_name}, tune_flags_int={self.default_x264_tune} (parsed from '{tune_names_str}')")
        logger.info(f"HLS Params: Target Duration: {self.hls_target_duration}s, Playlist Length: {self.hls_playlist_length} segments.")


    def _on_rtpbin_pad_added(self, rtpbin: Gst.Element, new_pad: Gst.Pad, _pipeline_ref: Gst.Pipeline):
        pad_name = new_pad.get_name()
        caps = new_pad.get_current_caps()
        if not caps: 
            logger.warning(f"Pad {pad_name} has no caps. Ignoring.")
            return
        struct = caps.get_structure(0)
        if not struct: 
            logger.warning(f"Pad {pad_name} caps have no structure. Ignoring.")
            return

        media_type = struct.get_string("media")
        encoding_name = struct.get_string("encoding-name")
        logger.debug(f"New pad added: {pad_name} with caps: {caps.to_string()}")

        if "recv_rtp_src" not in pad_name or media_type != "video" or encoding_name.upper() != "H264":
            logger.info(f"Ignoring pad {pad_name} (Media: {media_type}, Encoding: {encoding_name}) - not H264 video.")
            return

        ssrc_from_caps_tuple = struct.get_uint("ssrc")
        if not ssrc_from_caps_tuple[0]: 
            logger.warning(f"Could not get SSRC from caps for pad {pad_name}. Ignoring.")
            return
        ssrc = str(ssrc_from_caps_tuple[1])

        with self.lock:
            if ssrc not in self.whitelisted_ssrcs:
                logger.warning(f"SSRC {ssrc} received, but not in whitelist. Ignoring media.")
                return
            if ssrc in self.active_streams and self.active_streams[ssrc].get("status") != StreamState.REQUESTED:
                logger.info(f"Media for SSRC {ssrc} already being processed or HLS active. Pad: {pad_name}")
                return
            
            logger.info(f"Whitelisted SSRC {ssrc} detected. Setting up transcoding and HLS pipeline branch.")
            
            ssrc_hls_dir = self.hls_output_dir / ssrc
            ssrc_hls_dir.mkdir(parents=True, exist_ok=True)

            element_factories = {
                "rtph264depay": "rtph264depay", "h264parse_in": "h264parse", 
                "avdec_h264": "avdec_h264", "videoconvert": "videoconvert", 
                "videoscale": "videoscale", "videorate": "videorate",
                "transcode_capsfilter": "capsfilter", "x264enc": "x264enc",
                "h264parse_out": "h264parse", "hlssink": "hlssink2"
            }
            created_elements_dict = {}
            failed_elements_factories = []

            for el_name, factory_name in element_factories.items():
                element = Gst.ElementFactory.make(factory_name, f"{el_name.replace('_', '-')}-{ssrc}")
                if not element:
                    logger.error(f"SSRC {ssrc}: Failed to create GStreamer element '{el_name}' using factory '{factory_name}'. Check GStreamer plugin installation for '{factory_name}'.")
                    failed_elements_factories.append(factory_name)
                created_elements_dict[el_name] = element
            
            if failed_elements_factories:
                logger.error(f"SSRC {ssrc}: Aborting branch creation. Failed to create: {', '.join(set(failed_elements_factories))}.")
                self.active_streams[ssrc] = {"status": StreamState.ERROR, "pipeline_branch_elements": []}
                return

            rtph264depay = created_elements_dict["rtph264depay"]
            h264parse_in = created_elements_dict["h264parse_in"]
            avdec_h264 = created_elements_dict["avdec_h264"]
            videoconvert = created_elements_dict["videoconvert"]
            videoscale = created_elements_dict["videoscale"]
            videorate = created_elements_dict["videorate"]
            transcode_capsfilter = created_elements_dict["transcode_capsfilter"]
            x264enc = created_elements_dict["x264enc"]
            h264parse_out = created_elements_dict["h264parse_out"]
            hlssink = created_elements_dict["hlssink"]
            
            initial_caps_str = (
                f"video/x-raw,width={self.default_width},height={self.default_height},"
                f"framerate={self.default_framerate_num}/{self.default_framerate_den}"
            )
            initial_gst_caps = Gst.Caps.from_string(initial_caps_str)
            transcode_capsfilter.set_property("caps", initial_gst_caps)
            
            x264enc.set_property("bitrate", self.default_bitrate_kbps) 
            x264enc.set_property("speed-preset", self.default_x264_speed_preset)
            # The tune warning originates here. The self.default_x264_tune is an integer.
            # If the warning persists, it means the GObject layer doesn't like this integer for the GstX264EncTune GFlags type.
            x264enc.set_property("tune", self.default_x264_tune) 
            x264enc.set_property("threads", 0) 
            x264enc.set_property("byte-stream", True) 

            h264parse_out.set_property("config-interval", -1) 

            hls_params = {
                "target_duration": self.hls_target_duration,
                "playlist_length": self.hls_playlist_length,
                "playlist_root": f"{self.fastapi_base_url}/hls/{ssrc}",
                "playlist_location": str(ssrc_hls_dir / "playlist.m3u8"),
                "segment_location_pattern": str(ssrc_hls_dir / "segment%05d.ts")
            }
            hlssink.set_property("playlist_root", hls_params["playlist_root"])
            hlssink.set_property("playlist_location", hls_params["playlist_location"])
            hlssink.set_property("location", hls_params["segment_location_pattern"])
            hlssink.set_property("target-duration", hls_params["target_duration"])
            hlssink.set_property("max-files", hls_params["playlist_length"])

            pipeline_branch_elements = [
                rtph264depay, h264parse_in, avdec_h264, videoconvert, videoscale, videorate,
                transcode_capsfilter, x264enc, h264parse_out, hlssink
            ]
            transcoding_control_elements = { 
                "videoscale": videoscale, "videorate": videorate,
                "capsfilter": transcode_capsfilter, "x264enc": x264enc
            }

            if self.main_pipeline:
                for el in pipeline_branch_elements: 
                    if el: self.main_pipeline.add(el)
            else: 
                logger.error(f"Main pipeline not available to add HLS elements for SSRC {ssrc}.") # Corrected SSRC logging
                self.active_streams[ssrc] = {"status": StreamState.ERROR, "pipeline_branch_elements": []}
                return

            sink_pad_depay = rtph264depay.get_static_pad("sink")
            if not new_pad.link(sink_pad_depay) == Gst.PadLinkReturn.OK:
                logger.error(f"Failed to link rtpbin output pad to rtph264depay for SSRC {ssrc}.")
                if self.main_pipeline:
                    for el_to_remove in pipeline_branch_elements:
                        if el_to_remove: self.main_pipeline.remove(el_to_remove)
                self.active_streams[ssrc] = {"status": StreamState.ERROR, "pipeline_branch_elements": pipeline_branch_elements}
                return
            
            try:
                # MODIFICATION: Changed from Gst.Element.link_many to pairwise linking
                logger.debug(f"SSRC {ssrc}: Attempting to link elements pairwise.")
                for i in range(len(pipeline_branch_elements) - 1):
                    source_el = pipeline_branch_elements[i]
                    sink_el = pipeline_branch_elements[i+1]
                    if not source_el.link(sink_el):
                        raise Exception(f"Failed to link {source_el.get_name()} to {sink_el.get_name()}")
                logger.debug(f"SSRC {ssrc}: Pairwise linking successful.")
            except Exception as e:
                logger.error(f"Error linking elements in the new HLS branch for SSRC {ssrc}: {e}")
                if self.main_pipeline:
                    if new_pad.is_linked(): 
                        peer_pad = new_pad.get_peer()
                        if peer_pad: new_pad.unlink(peer_pad)
                    for el_to_remove in pipeline_branch_elements: 
                        if el_to_remove: self.main_pipeline.remove(el_to_remove)
                self.active_streams[ssrc] = {"status": StreamState.ERROR, "pipeline_branch_elements": pipeline_branch_elements}
                return

            for el in pipeline_branch_elements: 
                if el: el.sync_state_with_parent()

            current_transcode_params = {
                "bitrate_kbps": self.default_bitrate_kbps,
                "width": self.default_width, "height": self.default_height,
                "framerate_num": self.default_framerate_num, "framerate_den": self.default_framerate_den,
                "x264_speed_preset": x264enc.get_property("speed-preset"), 
                "x264_tune": x264enc.get_property("tune") # This will reflect actual value if set, or default if warning prevented set
            }

            self.active_streams[ssrc] = {
                "status": StreamState.RECEIVING_MEDIA,
                "pipeline_branch_elements": pipeline_branch_elements,
                "transcoding_control_elements": transcoding_control_elements,
                "current_transcode_params": current_transcode_params,
                "hls_params": hls_params,
                "hls_output_path": str(ssrc_hls_dir)
            }
            logger.info(f"Transcoding and HLS pipeline branch created for SSRC {ssrc}. Media received. Waiting for /start command.")


    def _on_bus_message(self, bus: Gst.Bus, message: Gst.Message):
        t = message.type
        src_element_name = "Unknown source"
        if message.src and hasattr(message.src, 'get_path_string'): 
            src_element_name = message.src.get_path_string()

        if t == Gst.MessageType.EOS:
            logger.info(f"GStreamer EOS received on bus from {src_element_name}.")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"GStreamer ERROR on bus from {src_element_name}: {err}, {debug}")
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning(f"GStreamer WARNING on bus from {src_element_name}: {warn}, {debug}")
        return True 

    def _build_pipeline(self):
        self.main_pipeline = Gst.Pipeline.new("rtp-hls-server-pipeline")
        rtp_udpsrc_caps_str = "application/x-rtp,media=(string)video,clock-rate=(int)90000,encoding-name=(string)H264"
        rtp_udpsrc_caps = Gst.Caps.from_string(rtp_udpsrc_caps_str)
        
        rtp_udpsrc = Gst.ElementFactory.make("udpsrc", "rtp-source")
        if not rtp_udpsrc: 
            logger.critical("Failed to create rtp_udpsrc (rtp-source) element. Check GStreamer installation.")
            return False
        rtp_udpsrc.set_property("port", self.rtp_port)
        rtp_udpsrc.set_property("caps", rtp_udpsrc_caps)

        rtcp_udpsrc = Gst.ElementFactory.make("udpsrc", "rtcp-source")
        if not rtcp_udpsrc: 
            logger.critical("Failed to create rtcp_udpsrc (rtcp-source) element.")
            return False
        rtcp_udpsrc.set_property("port", self.rtp_port + 1) 

        rtcp_udpsink = Gst.ElementFactory.make("udpsink", "rtcp-sink")
        if not rtcp_udpsink: 
            logger.critical("Failed to create rtcp_udpsink (rtcp-sink) element.")
            return False
        rtcp_udpsink.set_property("port", self.rtp_port + 2) 
        rtcp_udpsink.set_property("host", "127.0.0.1") 
        rtcp_udpsink.set_property("sync", False)
        rtcp_udpsink.set_property("async", False)

        self.rtpbin = Gst.ElementFactory.make("rtpbin", "rtpbin")
        if not self.rtpbin: 
            logger.critical("Failed to create rtpbin element.")
            return False
        self.rtpbin.set_property("latency", 200) 

        if not self.main_pipeline: 
             logger.critical("Main pipeline object is None before adding elements.")
             return False

        self.main_pipeline.add(rtp_udpsrc)
        self.main_pipeline.add(rtcp_udpsrc)
        self.main_pipeline.add(rtcp_udpsink)
        self.main_pipeline.add(self.rtpbin)

        if not rtp_udpsrc.link_pads(None, self.rtpbin, "recv_rtp_sink_0"):
            logger.critical("Failed to link RTP udpsrc to rtpbin.recv_rtp_sink_0")
            return False
        if not rtcp_udpsrc.link_pads(None, self.rtpbin, "recv_rtcp_sink_0"):
            logger.critical("Failed to link RTCP udpsrc to rtpbin.recv_rtcp_sink_0")
            return False
        if not self.rtpbin.link_pads("send_rtcp_src_0", rtcp_udpsink, "sink"):
            logger.critical("Failed to link rtpbin.send_rtcp_src_0 to RTCP udpsink")
            return False
        
        self.rtpbin.connect("pad-added", self._on_rtpbin_pad_added, self.main_pipeline)
        bus = self.main_pipeline.get_bus()
        if not bus: 
            logger.critical("Failed to get bus from main pipeline.")
            return False
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        logger.info("Main GStreamer pipeline built successfully.")
        return True

    def _run_gst_loop(self):
        self.gst_loop = GLib.MainLoop()
        logger.info(f"GStreamer server starting main loop. Listening for RTP on UDP:{self.rtp_port}")
        if not self.main_pipeline:
            logger.error("Cannot start GStreamer loop: main_pipeline is None.")
            if self.gst_loop and self.gst_loop.is_running(): self.gst_loop.quit() 
            return
        ret = self.main_pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("Unable to set the GStreamer pipeline to the PLAYING state.")
            if self.gst_loop and self.gst_loop.is_running(): self.gst_loop.quit()
            return
        try:
            self.gst_loop.run()
        except KeyboardInterrupt: logger.info("Keyboard interrupt in GStreamer loop.")
        except Exception as e: logger.error(f"GStreamer loop error: {e}", exc_info=True)
        finally:
            logger.info("GStreamer loop stopped.")
            if self.main_pipeline:
                 current_state_tuple = self.main_pipeline.get_state(Gst.CLOCK_TIME_NONE)
                 if current_state_tuple[0] == Gst.StateChangeReturn.SUCCESS and current_state_tuple[1] != Gst.State.NULL:
                    GLib.idle_add(self.main_pipeline.set_state, Gst.State.NULL)


    def start_server(self):
        if self.gst_thread and self.gst_thread.is_alive(): 
            logger.warning("GStreamer server already running.")
            return
        if not self._build_pipeline(): 
            logger.error("Failed to build GStreamer pipeline. Server not started.")
            return
        self.hls_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured HLS output directory exists: {self.hls_output_dir}")
        
        logger.info("Starting GStreamer thread...")
        self.gst_thread = threading.Thread(target=self._run_gst_loop, daemon=True)
        self.gst_thread.start()

    def stop_server(self):
        logger.info("Attempting to stop GStreamer server...")
        with self.lock:
            ssr_to_stop = list(self.active_streams.keys()) 
            for ssrc in ssr_to_stop:
                self._cleanup_ssrc_stream(ssrc, remove_from_whitelist=True) 
        
        if self.gst_loop and self.gst_loop.is_running():
            GLib.idle_add(self.gst_loop.quit)
        
        if self.gst_thread and self.gst_thread.is_alive():
            self.gst_thread.join(timeout=10) 
            if self.gst_thread.is_alive():
                logger.warning("GStreamer thread did not stop in time.")
        
        if self.main_pipeline:
            logger.info("Setting main GStreamer pipeline to NULL state.")
            current_state_result = self.main_pipeline.get_state(Gst.CLOCK_TIME_NONE) # result, state, pending
            if current_state_result[0] == Gst.StateChangeReturn.SUCCESS and current_state_result[1] != Gst.State.NULL:
                 self.main_pipeline.set_state(Gst.State.NULL)
            self.main_pipeline = None 
        
        self.gst_loop = None
        self.gst_thread = None
        self.whitelisted_ssrcs.clear()
        logger.info("GStreamer server stopped.")

    def request_ssrc(self) -> str:
        with self.lock:
            while True:
                new_ssrc = str(random.getrandbits(32))
                if new_ssrc not in self.whitelisted_ssrcs and new_ssrc not in self.active_streams:
                    self.whitelisted_ssrcs.add(new_ssrc)
                    self.active_streams[new_ssrc] = { 
                        "status": StreamState.REQUESTED,
                        "pipeline_branch_elements": [],
                        "transcoding_control_elements": {},
                        "current_transcode_params": { 
                            "bitrate_kbps": self.default_bitrate_kbps,
                            "width": self.default_width, "height": self.default_height,
                            "framerate_num": self.default_framerate_num, "framerate_den": self.default_framerate_den,
                            "x264_speed_preset": self.default_x264_speed_preset,
                            "x264_tune": self.default_x264_tune
                        },
                        "hls_params": {}, "hls_output_path": None
                    }
                    logger.info(f"SSRC {new_ssrc} requested and whitelisted.")
                    return new_ssrc

    def start_hls_stream(self, ssrc: str) -> Tuple[bool, str]:
        with self.lock:
            stream_info = self.active_streams.get(ssrc)
            if not stream_info: return False, "SSRC not found or not whitelisted."
            
            if stream_info["status"] == StreamState.STREAMING_HLS:
                return True, "HLS streaming already active."
            if stream_info["status"] == StreamState.REQUESTED:
                return False, "Media not yet received for this SSRC."
            if stream_info["status"] != StreamState.RECEIVING_MEDIA:
                 return False, f"Stream in unexpected state: {stream_info['status']}."

            stream_info["status"] = StreamState.STREAMING_HLS
            playlist_url_msg = f" Segments at {stream_info['hls_params']['playlist_root']}/playlist.m3u8" if stream_info.get('hls_params') and stream_info['hls_params'].get('playlist_root') else ""
            logger.info(f"HLS stream for SSRC {ssrc} officially started.{playlist_url_msg}")
            return True, f"HLS streaming started for SSRC {ssrc}."


    def _cleanup_ssrc_stream(self, ssrc: str, remove_from_whitelist: bool = True):
        stream_info = self.active_streams.pop(ssrc, None)
        if not stream_info:
            if remove_from_whitelist: self.whitelisted_ssrcs.discard(ssrc)
            return

        logger.info(f"Cleaning up stream for SSRC {ssrc}...")
        if self.main_pipeline and "pipeline_branch_elements" in stream_info:
            elements = stream_info["pipeline_branch_elements"]
            
            if elements and elements[0]: 
                depay_element = elements[0] # rtph264depay
                # Check if depay_element is not None and is part of the main pipeline before proceeding
                if depay_element and self.main_pipeline.get_by_name(depay_element.get_name()):
                    depay_sink_pad = depay_element.get_static_pad("sink")
                    if depay_sink_pad and depay_sink_pad.is_linked():
                        rtpbin_src_pad = depay_sink_pad.get_peer()
                        if rtpbin_src_pad:
                            logger.debug(f"SSRC {ssrc}: Unlinking {rtpbin_src_pad.get_name()} from {depay_sink_pad.get_name()}")
                            rtpbin_src_pad.unlink(depay_sink_pad)
                        else:
                            logger.warning(f"SSRC {ssrc}: Could not get peer pad for depayloader sink pad during cleanup.")
                    elif depay_sink_pad:
                        logger.debug(f"SSRC {ssrc}: Depayloader sink pad {depay_sink_pad.get_name()} was not linked.")
                else:
                    logger.warning(f"SSRC {ssrc}: Depayloader element not found in pipeline or is None during cleanup.")


            for el in reversed(elements):
                if el and hasattr(el, 'set_state') and self.main_pipeline.get_by_name(el.get_name()): 
                    el.set_state(Gst.State.NULL)
            
            def _remove_elements_from_pipeline_idle():
                logger.debug(f"SSRC {ssrc}: Removing elements from pipeline in idle: {[e.get_name() if e else 'None' for e in elements]}")
                all_removed = True
                for el_to_remove in reversed(elements):
                    if el_to_remove and self.main_pipeline and self.main_pipeline.get_by_name(el_to_remove.get_name()):
                        if not self.main_pipeline.remove(el_to_remove):
                            logger.warning(f"SSRC {ssrc}: Failed to remove element {el_to_remove.get_name()} from pipeline.")
                            all_removed = False
                if all_removed:
                    logger.debug(f"SSRC {ssrc}: All branch elements removed from pipeline.")
                return GLib.SOURCE_REMOVE 

            GLib.idle_add(_remove_elements_from_pipeline_idle)


        hls_output_path_str = stream_info.get("hls_output_path")
        if hls_output_path_str:
            hls_path = Path(hls_output_path_str)
            if hls_path.exists() and hls_path.is_dir():
                try: 
                    shutil.rmtree(hls_path)
                    logger.info(f"Removed HLS directory: {hls_path}")
                except OSError as e: logger.error(f"Error removing HLS dir {hls_path}: {e}")
        
        if remove_from_whitelist:
            self.whitelisted_ssrcs.discard(ssrc)
            logger.info(f"SSRC {ssrc} removed from whitelist.")
        
        logger.info(f"Cleanup for SSRC {ssrc} complete.")


    def stop_hls_stream(self, ssrc: str) -> Tuple[bool, str]:
        with self.lock:
            if ssrc not in self.active_streams: 
                if ssrc in self.whitelisted_ssrcs: 
                    self.whitelisted_ssrcs.discard(ssrc)
                    logger.info(f"SSRC {ssrc} was whitelisted but not active, removed from whitelist.")
                    return True, "SSRC was whitelisted but not active, removed from whitelist."
                return False, "SSRC not found."
            self._cleanup_ssrc_stream(ssrc, remove_from_whitelist=True)
            return True, f"HLS streaming stopped and resources cleaned for SSRC {ssrc}."

    def adjust_stream_parameters(self, ssrc: str, params: Dict[str, Any]) -> Tuple[bool, str]:
        with self.lock:
            stream_info = self.active_streams.get(ssrc)
            if not stream_info or stream_info.get("status") not in [StreamState.RECEIVING_MEDIA, StreamState.STREAMING_HLS]:
                return False, "SSRC not found or not in an adjustable state."

            control_elements = stream_info.get("transcoding_control_elements")
            if not control_elements:
                return False, f"Transcoding elements not found for SSRC {ssrc}, cannot adjust." # Corrected SSRC logging

            x264enc = control_elements.get("x264enc")
            transcode_capsfilter = control_elements.get("capsfilter")

            if not x264enc or not transcode_capsfilter:
                return False, f"Essential transcoding elements (x264enc, capsfilter) missing for SSRC {ssrc}." # Corrected SSRC logging

            current_params = stream_info.get("current_transcode_params", {}).copy()
            made_changes = False
            log_msgs = []

            if "bitrate_kbps" in params and params["bitrate_kbps"] != current_params.get("bitrate_kbps"):
                try:
                    new_bitrate = int(params["bitrate_kbps"])
                    if new_bitrate > 0:
                        logger.info(f"SSRC {ssrc}: Adjusting x264enc bitrate to {new_bitrate} kbps.")
                        x264enc.set_property("bitrate", new_bitrate)
                        current_params["bitrate_kbps"] = new_bitrate
                        made_changes = True
                        log_msgs.append(f"Bitrate set to {new_bitrate}kbps.")
                    else: log_msgs.append(f"Invalid bitrate: {new_bitrate}.")
                except ValueError: log_msgs.append(f"Invalid bitrate value type: {params['bitrate_kbps']}.")
            
            new_width = current_params.get("width")
            new_height = current_params.get("height")
            new_framerate_num = current_params.get("framerate_num")
            new_framerate_den = current_params.get("framerate_den")
            caps_changed = False

            if "resolution_width" in params and params["resolution_width"] != new_width:
                try: new_width = int(params["resolution_width"]); caps_changed = True
                except ValueError: log_msgs.append("Invalid width value.")
            if "resolution_height" in params and params["resolution_height"] != new_height:
                try: new_height = int(params["resolution_height"]); caps_changed = True
                except ValueError: log_msgs.append("Invalid height value.")
            if "framerate_numerator" in params and params["framerate_numerator"] != new_framerate_num:
                try: new_framerate_num = int(params["framerate_numerator"]); caps_changed = True
                except ValueError: log_msgs.append("Invalid framerate numerator value.")
            if "framerate_denominator" in params and params["framerate_denominator"] != new_framerate_den:
                try: new_framerate_den = int(params["framerate_denominator"]); caps_changed = True
                except ValueError: log_msgs.append("Invalid framerate denominator value.")

            if caps_changed:
                if not (new_width > 0 and new_height > 0 and new_framerate_num > 0 and new_framerate_den > 0):
                    log_msgs.append("Invalid resolution/framerate values for caps update. All must be positive.")
                else:
                    new_caps_str = f"video/x-raw,width={new_width},height={new_height},framerate={new_framerate_num}/{new_framerate_den}"
                    new_gst_caps = Gst.Caps.from_string(new_caps_str)
                    logger.info(f"SSRC {ssrc}: Attempting to adjust capsfilter to: {new_caps_str}")
                    transcode_capsfilter.set_property("caps", new_gst_caps)
                    
                    current_params["width"] = new_width
                    current_params["height"] = new_height
                    current_params["framerate_num"] = new_framerate_num
                    current_params["framerate_den"] = new_framerate_den
                    made_changes = True
                    log_msgs.append(f"Resolution/Framerate target updated to {new_width}x{new_height}@{new_framerate_num}/{new_framerate_den}fps. Effect depends on pipeline's ability to renegotiate live.")
            
            if made_changes:
                stream_info["current_transcode_params"] = current_params
                final_message = "Adjustment parameters applied. " + " ".join(log_msgs)
                logger.info(f"SSRC {ssrc}: {final_message}")
                return True, final_message
            
            final_message = "No valid or changed parameters provided. " + " ".join(log_msgs)
            logger.info(f"SSRC {ssrc}: {final_message}") # Log even if no changes made but params were present
            return False, final_message


    def get_stream_status(self, ssrc: str) -> Optional[Dict[str, Any]]:
        with self.lock:
            stream_info = self.active_streams.get(ssrc)
            if stream_info:
                playlist_url = None
                hls_params = stream_info.get("hls_params", {})
                if hls_params and stream_info.get("status") == StreamState.STREAMING_HLS:
                    playlist_root = hls_params.get("playlist_root")
                    if playlist_root: playlist_url = f"{playlist_root}/playlist.m3u8"
                
                return {
                    "ssrc": ssrc,
                    "status": stream_info.get("status"),
                    "hls_params": hls_params,
                    "hls_playlist_url": playlist_url,
                    "transcoding_params": stream_info.get("current_transcode_params")
                }
            return None

    def get_system_analytics(self) -> Dict[str, Any]:
        with self.lock:
            active_hls_streams = [ssrc for ssrc, info in self.active_streams.items() if info.get("status") == StreamState.STREAMING_HLS]
            return {
                "total_whitelisted_ssrcs": len(self.whitelisted_ssrcs),
                "total_active_streams_tracked": len(self.active_streams), 
                "count_streaming_hls": len(active_hls_streams),
                "ssrcs_streaming_hls": active_hls_streams,
            }

# --- FastAPI Application ---
gstreamer_server_instance: Optional[GStreamerServer] = None
HLS_OUTPUT_DIR_GLOBAL: Optional[str] = None
HLS_PROCESSED_OUTPUT_DIR_GLOBAL: Optional[str] = None
cli_args_global: Optional[argparse.Namespace] = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global gstreamer_server_instance, HLS_OUTPUT_DIR_GLOBAL, HLS_PROCESSED_OUTPUT_DIR_GLOBAL, cli_args_global, config

    if DEFAULT_CONFIG_PATH.exists():
        config.read(DEFAULT_CONFIG_PATH)
        logger.info(f"Loaded configuration from {DEFAULT_CONFIG_PATH}")
    else:
        logger.warning(f"Configuration file {DEFAULT_CONFIG_PATH} not found. Using hardcoded defaults and populating config object.")
        for section, settings in DEFAULT_SETTINGS.items():
            if not config.has_section(section): config.add_section(section)
            for key, value in settings.items():
                if not config.has_option(section, key): config.set(section, key, value)

    logger.info("FastAPI application starting up...")
    
    rtp_port = cli_args_global.rtp_port if cli_args_global and cli_args_global.rtp_port is not None else get_config_value("GStreamer", "rtp_port", int)
    hls_dir = cli_args_global.hls_output_dir if cli_args_global and cli_args_global.hls_output_dir else get_config_value("GStreamer", "hls_output_dir", str)
    hls_proc_dir = cli_args_global.hls_processed_output_dir if cli_args_global and cli_args_global.hls_processed_output_dir else get_config_value("GStreamer", "hls_processed_output_dir", str)
    fastapi_host = cli_args_global.host if cli_args_global and cli_args_global.host else get_config_value("FastAPI", "host", str)
    fastapi_port_num = cli_args_global.fastapi_port if cli_args_global and cli_args_global.fastapi_port is not None else get_config_value("FastAPI", "port", int)

    HLS_OUTPUT_DIR_GLOBAL = hls_dir
    HLS_PROCESSED_OUTPUT_DIR_GLOBAL = hls_proc_dir

    gstreamer_server_instance = GStreamerServer(
        rtp_port=rtp_port, hls_output_dir=hls_dir,
        fastapi_host=fastapi_host, fastapi_port=fastapi_port_num,
        app_config=config 
    )
    gstreamer_server_instance.start_server()
    
    hls_static_path = Path(HLS_OUTPUT_DIR_GLOBAL)
    hls_static_path.mkdir(parents=True, exist_ok=True) 
    app.mount("/hls", StaticFiles(directory=hls_static_path, check_dir=False), name="hls") 
    logger.info(f"Mounted /hls to serve files from {hls_static_path.resolve()}")

    hls_proc_static_path = Path(HLS_PROCESSED_OUTPUT_DIR_GLOBAL)
    hls_proc_static_path.mkdir(parents=True, exist_ok=True)
    app.mount("/processed_hls", StaticFiles(directory=hls_proc_static_path, check_dir=False), name="processed_hls")
    logger.info(f"Mounted /processed_hls to serve files from {hls_proc_static_path.resolve()}")

    yield
    logger.info("FastAPI application shutting down...")
    if gstreamer_server_instance:
        gstreamer_server_instance.stop_server()

app = FastAPI(lifespan=lifespan)

# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def get_index_page(request: Request):
    if not gstreamer_server_instance:
        return HTMLResponse("<html><body style='font-family: sans-serif; padding: 20px; text-align: center;'><h1>Server Error</h1><p>GStreamer server instance is not available.</p></body></html>", status_code=503)

    active_hls_streams_info = []
    if gstreamer_server_instance: 
        with gstreamer_server_instance.lock: 
            stream_order = 0
            for ssrc, info_dict in gstreamer_server_instance.active_streams.items(): 
                if info_dict.get("status") == StreamState.STREAMING_HLS and info_dict.get("hls_params"):
                    stream_order += 1
                    playlist_root_url = info_dict["hls_params"].get('playlist_root', '')
                    playlist_url = f"{playlist_root_url}/playlist.m3u8" if playlist_root_url else "#"
                    trans_params = info_dict.get("current_transcode_params", {})
                    trans_details = (
                        f"{trans_params.get('width','N/A')}x{trans_params.get('height','N/A')}@ "
                        f"{trans_params.get('framerate_num','N/A')}/{trans_params.get('framerate_den','N/A')}fps, "
                        f"{trans_params.get('bitrate_kbps','N/A')}kbps"
                    )

                    processed_playlist_url = f"{gstreamer_server_instance.fastapi_base_url}/processed_hls/{ssrc}/playlist.m3u8"
                    
                    active_hls_streams_info.append({
                        "ssrc": ssrc,
                        "playlist_url": playlist_url,
                        "processed_playlist_url": processed_playlist_url,
                        "order": stream_order,
                        "transcoding_details": trans_details
                    })
    
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Active HLS Streams (Transcoded)</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
        <style> body { font-family: 'Inter', sans-serif; } .stream-card { transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out; } .stream-card:hover { transform: translateY(-5px); box-shadow: 0 10px 20px rgba(0,0,0,0.15); } </style>
    </head>
    <body class="bg-gray-100 min-h-screen flex flex-col items-center py-10 px-4">
        <div class="w-full max-w-5xl">
            <header class="mb-10 text-center"><h1 class="text-4xl font-bold text-gray-800">Active HLS Streams</h1></header>
            <main id="streams-list" class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-8">
    """
    if not active_hls_streams_info:
        html_content += "<p class='text-gray-600 col-span-full text-center text-lg py-10'>No active HLS streams. Use API to request SSRC and start.</p>"
    else:
        for stream_card_info in active_hls_streams_info: 
            html_content += f"""
            <div class="stream-card bg-white rounded-xl shadow-lg p-6">
                <h3 class="text-xl font-semibold text-gray-700 mb-2">Stream {stream_card_info['order']}</h3>
                <p class="text-sm text-gray-500 mb-1">SSRC: <span class="font-medium text-gray-600">{stream_card_info['ssrc']}</span></p>
                <p class="text-sm text-gray-500 mb-1">Params: <span class="font-medium text-gray-600 break-all">{stream_card_info['transcoding_details']}</span></p>
                <p class="text-sm text-gray-500 mb-3">Status: <span class="font-semibold text-green-600">Streaming HLS</span></p>
                <a href="{stream_card_info['playlist_url']}" target="_blank" class="text-xs text-blue-600 hover:text-blue-800 break-all block mb-4">{stream_card_info['playlist_url']}</a>
                <button onclick="toggleStreams('{stream_card_info['ssrc']}', '{stream_card_info['playlist_url']}', '{stream_card_info['processed_playlist_url']}')" id="playBtn-{stream_card_info['ssrc']}"
                        class="w-full bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded-lg transition duration-150 ease-in-out focus:outline-none focus:ring-2 focus:ring-blue-400 focus:ring-opacity-75">
                    Play Stream {stream_card_info['order']}
                </button>
                <div class="mt-4 rounded-lg overflow-hidden aspect-video bg-black"><video id="video-{stream_card_info['ssrc']}" class="w-full h-full" controls style="display:none;"></video></div>
                <div class="mt-4 rounded-lg overflow-hidden aspect-video bg-black"><video id="video-proc-{stream_card_info['ssrc']}" class="w-full h-full" controls style="display:none;"></video></div>
            </div>"""
    html_content += """
            </main>
        </div>
        <script>
            const hlsInstances = {};
            function togglePlayStream(ssrc, playlistUrl) {
                const videoElement = document.getElementById('video-' + ssrc);
                const playButton = document.getElementById('playBtn-' + ssrc);
                const streamOrderMatch = playButton.textContent.match(/\\d+$/);
                const streamOrder = streamOrderMatch ? streamOrderMatch[0] : '';

                if (hlsInstances[ssrc]) {
                    hlsInstances[ssrc].destroy(); delete hlsInstances[ssrc];
                    videoElement.style.display = 'none'; videoElement.removeAttribute('src');  videoElement.load(); 
                    playButton.textContent = 'Play Stream ' + streamOrder; return;
                }
                videoElement.style.display = 'block'; playButton.textContent = 'Stop Stream ' + streamOrder;
                if (Hls.isSupported()) {
                    const hls = new Hls({ enableWorker: true, lowLatencyMode: true, fragLoadingMaxRetry: 4, manifestLoadingMaxRetry: 2, defaultAudioCodec: 'mp4a.40.2' });
                    hls.loadSource(playlistUrl); hls.attachMedia(videoElement);
                    hls.on(Hls.Events.MANIFEST_PARSED, () => videoElement.play().catch(e => console.error('Play error for SSRC ' + ssrc + ':', e)));
                    hls.on(Hls.Events.ERROR, (event, data) => {
                        console.error('HLS.js error for SSRC ' + ssrc + ':', event, data);
                        if (data.fatal) { 
                            switch(data.type) {
                                case Hls.ErrorTypes.NETWORK_ERROR: hls.startLoad(); break; 
                                case Hls.ErrorTypes.MEDIA_ERROR: hls.recoverMediaError(); break; 
                                default: hls.destroy(); delete hlsInstances[ssrc]; videoElement.style.display = 'none'; playButton.textContent = 'Play Stream ' + streamOrder; break;
                            }
                        }
                    });
                    hlsInstances[ssrc] = hls;
                } else if (videoElement.canPlayType('application/vnd.apple.mpegurl')) {
                    videoElement.src = playlistUrl;
                    videoElement.addEventListener('loadedmetadata', () => videoElement.play().catch(e => console.error('Native HLS play error for SSRC ' + ssrc + ':', e)));
                    hlsInstances[ssrc] = { destroy: () => { videoElement.pause(); videoElement.removeAttribute('src'); videoElement.load(); } };
                } else {
                    alert('HLS playback not supported in this browser.'); videoElement.style.display = 'none'; playButton.textContent = 'Play Stream ' + streamOrder;
                }
            };

            const hlsProcInstances = {};
            function togglePlayProcStream(ssrc, playlistUrl) {
                const videoElement = document.getElementById('video-proc-' + ssrc);

                if (hlsProcInstances[ssrc]) {
                    hlsProcInstances[ssrc].destroy(); delete hlsProcInstances[ssrc];
                    videoElement.style.display = 'none'; videoElement.removeAttribute('src');  videoElement.load(); 
                    return;
                }
                videoElement.style.display = 'block'; 
                if (Hls.isSupported()) {
                    const hls = new Hls({ enableWorker: true, lowLatencyMode: true, fragLoadingMaxRetry: 4, manifestLoadingMaxRetry: 2, defaultAudioCodec: 'mp4a.40.2' });
                    hls.loadSource(playlistUrl); hls.attachMedia(videoElement);
                    hls.on(Hls.Events.MANIFEST_PARSED, () => videoElement.play().catch(e => console.error('Play error for SSRC ' + ssrc + ':', e)));
                    hls.on(Hls.Events.ERROR, (event, data) => {
                        console.error('HLS.js error for SSRC ' + ssrc + ':', event, data);
                        if (data.fatal) { 
                            switch(data.type) {
                                case Hls.ErrorTypes.NETWORK_ERROR: hls.startLoad(); break; 
                                case Hls.ErrorTypes.MEDIA_ERROR: hls.recoverMediaError(); break; 
                                default: hls.destroy(); delete hlsProcInstances[ssrc]; videoElement.style.display = 'none'; break;
                            }
                        }
                    });
                    hlsProcInstances[ssrc] = hls;
                } else if (videoElement.canPlayType('application/vnd.apple.mpegurl')) {
                    videoElement.src = playlistUrl;
                    videoElement.addEventListener('loadedmetadata', () => videoElement.play().catch(e => console.error('Native HLS play error for SSRC ' + ssrc + ':', e)));
                    hlsProcInstances[ssrc] = { destroy: () => { videoElement.pause(); videoElement.removeAttribute('src'); videoElement.load(); } };
                } else {
                    alert('HLS playback not supported in this browser.'); videoElement.style.display = 'none';
                }
            };

            function toggleStreams(ssrc, playlistUrl, procPlaylistUrl) {
                togglePlayStream(ssrc, playlistUrl);
                togglePlayProcStream(ssrc, procPlaylistUrl);
                return;
            }
        </script>
    </body></html>"""
    return HTMLResponse(content=html_content)


@app.post("/streams/request_ssrc")
async def request_ssrc_endpoint():
    if not gstreamer_server_instance: raise HTTPException(status_code=503, detail="GStreamer server not initialized.")
    ssrc = gstreamer_server_instance.request_ssrc()
    return {"ssrc": ssrc, "message": "SSRC whitelisted. Awaiting media. Default transcoding params will be applied."}

@app.post("/streams/start")
async def start_stream_endpoint(ssrc: str):
    if not gstreamer_server_instance: raise HTTPException(status_code=503, detail="GStreamer server not initialized.")
    success, message = gstreamer_server_instance.start_hls_stream(ssrc)
    if not success:
        if "not whitelisted" in message.lower() or "not found" in message.lower(): raise HTTPException(status_code=404, detail=message) 
        if "media not yet received" in message.lower(): raise HTTPException(status_code=425, detail=message) 
        raise HTTPException(status_code=400, detail=message)
    return {"ssrc": ssrc, "message": message, "status_details": gstreamer_server_instance.get_stream_status(ssrc)}

@app.post("/streams/stop")
async def stop_stream_endpoint(ssrc: str):
    if not gstreamer_server_instance: raise HTTPException(status_code=503, detail="GStreamer server not initialized.")
    success, message = gstreamer_server_instance.stop_hls_stream(ssrc)
    if not success: raise HTTPException(status_code=404, detail=message) 
    return {"ssrc": ssrc, "message": message}

@app.post("/streams/adjust") 
async def adjust_stream_parameters_endpoint(
    ssrc: str, 
    bitrate_kbps: Optional[int] = None,
    resolution_width: Optional[int] = None,
    resolution_height: Optional[int] = None,
    framerate_numerator: Optional[int] = None,
    framerate_denominator: Optional[int] = None
):
    if not gstreamer_server_instance:
        raise HTTPException(status_code=503, detail="GStreamer server not initialized.")
    
    adjust_params = {}
    if bitrate_kbps is not None:
        if not (isinstance(bitrate_kbps, int) and bitrate_kbps > 0):
            raise HTTPException(status_code=400, detail="Invalid bitrate_kbps. Must be positive integer.")
        adjust_params["bitrate_kbps"] = bitrate_kbps
    if resolution_width is not None:
        if not (isinstance(resolution_width, int) and resolution_width > 0):
            raise HTTPException(status_code=400, detail="Invalid resolution_width.")
        adjust_params["resolution_width"] = resolution_width
    if resolution_height is not None:
        if not (isinstance(resolution_height, int) and resolution_height > 0):
            raise HTTPException(status_code=400, detail="Invalid resolution_height.")
        adjust_params["resolution_height"] = resolution_height
    if framerate_numerator is not None:
        if not (isinstance(framerate_numerator, int) and framerate_numerator > 0):
            raise HTTPException(status_code=400, detail="Invalid framerate_numerator.")
        adjust_params["framerate_numerator"] = framerate_numerator
    if framerate_denominator is not None:
        if not (isinstance(framerate_denominator, int) and framerate_denominator > 0):
            raise HTTPException(status_code=400, detail="Invalid framerate_denominator.")
        adjust_params["framerate_denominator"] = framerate_denominator

    if not adjust_params:
        raise HTTPException(status_code=400, detail="No valid adjustment parameters provided.")

    success, message = gstreamer_server_instance.adjust_stream_parameters(ssrc, adjust_params)
    if not success:
        if "not found" in message.lower() or "not in an adjustable state" in message.lower():
             raise HTTPException(status_code=404, detail=message)
        raise HTTPException(status_code=400, detail=message) 
    
    return {"ssrc": ssrc, "message": message, "updated_params_status": gstreamer_server_instance.get_stream_status(ssrc)}


@app.get("/streams/status")
async def get_status_endpoint(ssrc: Optional[str] = None):
    if not gstreamer_server_instance: raise HTTPException(status_code=503, detail="GStreamer server not initialized.")
    if ssrc:
        status = gstreamer_server_instance.get_stream_status(ssrc)
        if not status: raise HTTPException(status_code=404, detail=f"SSRC {ssrc} not found.")
        return status
    else:
        return gstreamer_server_instance.get_system_analytics()

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastAPI server with GStreamer for HLS Transcoding and Restreaming.")
    parser.add_argument("-p", "--rtp-port", type=int, help="RTP port for GStreamer. Overrides config.")
    parser.add_argument("--host", type=str, help="Host for FastAPI. Overrides config.")
    parser.add_argument("--fastapi-port", type=int, help="Port for FastAPI. Overrides config.")
    parser.add_argument("--hls-output-dir", type=str, help="Directory for HLS files. Overrides config.")
    parser.add_argument("--hls_processed_output_dir", type=str, help="Directory for processed HLS files. Overrides config.")
    
    cli_args_global = parser.parse_args() 
    
    temp_config = configparser.ConfigParser()
    for section, settings in DEFAULT_SETTINGS.items():
        if not temp_config.has_section(section): temp_config.add_section(section)
        for key, value in settings.items():
            if not temp_config.has_option(section, key): temp_config.set(section, key, str(value)) 
            
    if DEFAULT_CONFIG_PATH.exists(): temp_config.read(DEFAULT_CONFIG_PATH)

    uvicorn_host = cli_args_global.host if cli_args_global.host else temp_config.get("FastAPI", "host")
    uvicorn_port = cli_args_global.fastapi_port if cli_args_global.fastapi_port is not None else temp_config.getint("FastAPI", "port")

    logger.info(f"FastAPI will attempt to run on http://{uvicorn_host}:{uvicorn_port}")

    uvicorn.run(app, host=uvicorn_host, port=uvicorn_port, log_level="info")
