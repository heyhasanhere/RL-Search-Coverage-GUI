// Offscreen capture for players without an X server (macOS, Windows; sim/flooduav/README.md).
// The split views render into one texture instead of the screen, and its pixels go to a frame
// file that the bridge reads, so the window size and what covers it do not matter.
//
// Environment variables (optional):
//   FLOODUAV_CAPTURE      path of the frame file; unset: the views render to the screen (Xvfb and x11grab)
//   FLOODUAV_CAPTURE_FPS  frames a second (default 10)
//
// Frame file: a 32-byte header (int64 frame number, int32 width, int32 height, int32 slot), then two
// slots of width x height RGBA pixels, bottom row first. The next frame goes into the slot the header
// does not name, then the header names it; a reader whose frame number moved on by 2 or more re-reads.

using System;
using System.IO;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace FloodPatch
{
    public class Capture : MonoBehaviour
    {
        // The texture the split views render into; null: they render to the screen.
        public static RenderTexture Target;
        static FileStream file;
        static byte[] pixels;
        static readonly byte[] header = new byte[32];
        long frame;
        float period, next;
        bool pending;

        public static RenderTexture Setup(int width, int height)
        {
            string path = Environment.GetEnvironmentVariable("FLOODUAV_CAPTURE");
            if (string.IsNullOrEmpty(path) || Target != null)
                return Target;
            Target = new RenderTexture(width, height, 24, RenderTextureFormat.ARGB32);
            Target.name = "FloodPatchCapture";
            Target.Create();
            file = new FileStream(path, FileMode.Create, FileAccess.ReadWrite, FileShare.ReadWrite);
            file.SetLength(header.Length + 2L * width * height * 4);
            GameObject go = new GameObject("FloodPatchCapture");
            DontDestroyOnLoad(go);
            string s = Environment.GetEnvironmentVariable("FLOODUAV_CAPTURE_FPS");
            float fps;
            if (!float.TryParse(s, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out fps) || fps <= 0f)
                fps = 10f;
            go.AddComponent<Capture>().period = 1f / fps;
            Debug.Log("[FloodPatch] offscreen capture " + width + "x" + height + " at " + fps + " fps to " + path);
            return Target;
        }

        void LateUpdate()
        {
            if (pending || Time.unscaledTime < next)
                return;
            next = Time.unscaledTime + period;
            pending = true;
            AsyncGPUReadback.Request(Target, 0, TextureFormat.RGBA32, Done);
        }

        // The player window shows the views too, so it is clear what it is.
        void OnGUI()
        {
            if (Event.current.type != EventType.Repaint)
                return;
            float h = Screen.width * Target.height / (float)Target.width;
            GUI.DrawTexture(new Rect(0f, 0f, Screen.width, h), Target, ScaleMode.StretchToFill, false);
            GUI.Label(new Rect(6f, h + 2f, Screen.width - 12f, 40f), "FloodUAV simulation for the operator UI. Do not close; stop it with sim/flooduav/run.sh down.");
        }

        void Done(AsyncGPUReadbackRequest req)
        {
            pending = false;
            if (req.hasError)
                return;
            NativeArray<byte> data = req.GetData<byte>();
            if (pixels == null || pixels.Length != data.Length)
                pixels = new byte[data.Length];
            data.CopyTo(pixels);
            int slot = (int)((frame + 1) % 2);
            file.Seek(header.Length + (long)slot * pixels.Length, SeekOrigin.Begin);
            file.Write(pixels, 0, pixels.Length);
            frame++;
            Array.Copy(BitConverter.GetBytes(frame), 0, header, 0, 8);
            Array.Copy(BitConverter.GetBytes(Target.width), 0, header, 8, 4);
            Array.Copy(BitConverter.GetBytes(Target.height), 0, header, 12, 4);
            Array.Copy(BitConverter.GetBytes(slot), 0, header, 16, 4);
            file.Seek(0, SeekOrigin.Begin);
            file.Write(header, 0, header.Length);
            file.Flush();
        }
    }
}
