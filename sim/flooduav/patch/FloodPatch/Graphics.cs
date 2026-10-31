// Graphics presets from the bridge (Fleet message [8, render scale, msaa, fps]; sim/flooduav/README.md).
// The builds have one quality level (PC), so a preset sets the render pipeline instead: the share of the
// view size it renders (then scales up) and its anti-aliasing samples (1: off); the offscreen capture
// (Capture.cs) runs at `fps`.

using System.Reflection;
using UnityEngine;
using UnityEngine.Rendering;

namespace FloodPatch
{
    public static class Graphics
    {
        public static void Apply(float renderScale, int msaa, float fps)
        {
            object rp = QualitySettings.renderPipeline ?? GraphicsSettings.defaultRenderPipeline;
            PropertyInfo samples = rp == null ? null : rp.GetType().GetProperty("msaaSampleCount");
            PropertyInfo scale = rp == null ? null : rp.GetType().GetProperty("renderScale");
            if (samples != null && samples.CanWrite)
                samples.SetValue(rp, Mathf.Max(msaa, 1));
            if (scale != null && scale.CanWrite)
                scale.SetValue(rp, Mathf.Clamp(renderScale, 0.1f, 2f));
            Capture.SetFps(fps);
            Debug.Log("[FloodPatch] graphics: quality " + QualitySettings.names[QualitySettings.GetQualityLevel()]
                + ", msaa " + (samples == null ? "?" : samples.GetValue(rp)) + ", render scale " + (scale == null ? "?" : scale.GetValue(rp))
                + ", capture " + fps + " fps");
        }
    }
}
