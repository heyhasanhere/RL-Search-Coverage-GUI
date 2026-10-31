// Runtime hooks injected into the FloodUAV build (sim/flooduav/README.md).
// The injector adds a call to Hook.Drone at the start of DroneVelocityController.OnEnable
// and to Hook.Mission at the start of SearchMissionManager.BeginMission.
// Everything goes through reflection, so this assembly does not reference Assembly-CSharp.
//
// Environment variables (all optional):
//   FLOODUAV_HSPEED    max horizontal speed, m/s (build default 5)
//   FLOODUAV_VSPEED    max vertical speed, m/s (build default 2)
//   FLOODUAV_TILT      max tilt, degrees, 1..25 (build default 12); sets the horizontal acceleration
//   FLOODUAV_AIRCRAFT  comma-separated name=value float fields of CargoAircraftController
//   FLOODUAV_TIME_LIMIT mission time limit, s
//   FLOODUAV_SPLIT     1: split the 3920x720 screen (Layout below), or the offscreen texture of Capture.cs
//   FLOODUAV_INFO      path of a JSON file with the boundary, start points, survivors and camera field of view
//   FLOODUAV_FOLLOW    follow camera offset "x,y,z" behind the drone (build default 0,12,-40)
//   FLOODUAV_FLEET     1: the pickup drone becomes a second drone the bridge flies (Fleet.cs)
//   FLOODUAV_SOFT      1: contact with the ground or buildings no longer ends the mission, so drones can land
//   FLOODUAV_MAP       path the high resolution top-down map is written to (JPEG, Fleet message 5)
//   FLOODUAV_SURVIVOR_SEED  survivor layout seed (Layout.cs; default -1: the build's fixed survivors)
//   FLOODUAV_SURVIVOR_COUNT number of survivors (Layout.cs; default -1: the build's 15)
//   FLOODUAV_MAX_DT    game time one frame may advance, s (default 0.05). Unity waits for the bridge inside a
//                      frame, and that wait counts as frame time, so without a small cap one frame runs
//                      several bridge steps and their waits, and the frame rate falls to a few fps.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Reflection;
using System.Text;
using UnityEngine;

namespace FloodPatch
{
    public static class Hook
    {
        const BindingFlags F = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
        static bool viewsReady;
        static MonoBehaviour pickup;
        static Vector3[] pads;
        // the on-screen drone cameras (0 search drone, 1 pickup drone), their straight-down pose, and the overview
        static readonly Camera[] droneCams = new Camera[2];
        static readonly Quaternion[] camBase = new Quaternion[2];
        static Camera overview;
        // the follow cameras (CargoAircraftCamera): 0 the main camera behind the search drone, 1 a copy behind the pickup drone
        static readonly Component[] followCams = new Component[2];
        // each drone's own flight limits, read before the overrides: horizontal, vertical speed, tilt, wing area
        internal static readonly Dictionary<MonoBehaviour, float[]> BuildLimits = new Dictionary<MonoBehaviour, float[]>();

        // Screen layout in pixels (x, y from the top, w, h); the bridge crops the same regions.
        const float SW = 3920f, SH = 720f;
        static readonly float[] Follow = { 0, 0, 1280, 720 }, Cam0 = { 1280, 0, 640, 360 },
            Cam1 = { 1280, 360, 640, 360 }, Overview = { 1920, 0, 720, 720 }, Follow1 = { 2640, 0, 1280, 720 };

        static Rect R(float[] r)
        {
            return new Rect(r[0] / SW, 1f - (r[1] + r[3]) / SH, r[2] / SW, r[3] / SH);
        }

        static float Env(string key, float fallback)
        {
            string s = Environment.GetEnvironmentVariable(key);
            float v;
            return s != null && float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out v) ? v : fallback;
        }

        internal static object Get(object o, string field)
        {
            FieldInfo f = o.GetType().GetField(field, F);
            return f == null ? null : f.GetValue(o);
        }

        static void Set(object o, string field, string env)
        {
            FieldInfo f = o.GetType().GetField(field, F);
            if (f == null || Environment.GetEnvironmentVariable(env) == null)
                return;
            float v = Env(env, (float)f.GetValue(o));
            f.SetValue(o, v);
            Debug.Log("[FloodPatch] " + field + " = " + v.ToString(CultureInfo.InvariantCulture));
        }

        public static void Drone(MonoBehaviour controller)
        {
            Component wing = controller.GetComponent("CargoAircraftController");
            if (!BuildLimits.ContainsKey(controller))
                BuildLimits[controller] = new[] {
                    (float)Get(controller, "maximumHorizontalSpeed"), (float)Get(controller, "maximumVerticalSpeed"),
                    (float)Get(controller, "maximumTiltDegrees"), wing == null ? 0f : (float)Get(wing, "wingArea") };
            Set(controller, "maximumHorizontalSpeed", "FLOODUAV_HSPEED");
            Set(controller, "maximumVerticalSpeed", "FLOODUAV_VSPEED");
            Set(controller, "maximumTiltDegrees", "FLOODUAV_TILT");
            // The airframe drag (wing area times drag coefficient) caps the hover-mode speed
            // near 2.5 m/s whatever the commanded speed; FLOODUAV_AIRCRAFT="wingArea=2,sideDrag=0.1"
            Component aircraft = controller.GetComponent("CargoAircraftController");
            if (aircraft == null)
                return;
            string spec = Environment.GetEnvironmentVariable("FLOODUAV_AIRCRAFT") ?? "";
            foreach (string kv in spec.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries))
            {
                string[] p = kv.Split('=');
                FieldInfo f = aircraft.GetType().GetField(p[0].Trim(), F);
                float v;
                if (p.Length == 2 && f != null && f.FieldType == typeof(float)
                    && float.TryParse(p[1], NumberStyles.Float, CultureInfo.InvariantCulture, out v))
                    f.SetValue(aircraft, v);
            }
            StringBuilder sb = new StringBuilder("[FloodPatch] aircraft");
            foreach (string name in new[] { "mass", "wingArea", "maximumThrust", "parasiticDrag", "inducedDrag", "sideDrag", "liftAtZeroAngle" })
                sb.Append(' ').Append(name).Append('=').Append(((float)Get(aircraft, name)).ToString(CultureInfo.InvariantCulture));
            Debug.Log(sb.ToString());
        }

        // MissionMapDisplay.Start activates a second display when one exists, which opens a
        // full-screen window on macOS. Skip it, so the monitor setup does not matter.
        public static bool MapDisplay(MonoBehaviour display)
        {
            display.enabled = false;
            Debug.Log("[FloodPatch] mission map display off");
            return true;
        }

        public static void Mission(MonoBehaviour mission)
        {
            Application.runInBackground = true;
            Time.maximumDeltaTime = Mathf.Max(Env("FLOODUAV_MAX_DT", 0.05f), Time.fixedDeltaTime);
            Set(mission, "timeLimitSeconds", "FLOODUAV_TIME_LIMIT");
            try
            {
                if (!viewsReady)
                {
                    pickup = FindPickup(mission);
                    // both drones still sit on their pads at the first mission start
                    pads = new[] { ((Transform)Get(mission, "droneStartPoint")).position,
                                   pickup != null ? pickup.transform.position : ((Transform)Get(mission, "droneStartPoint")).position };
                    if (Env("FLOODUAV_SOFT", 0f) > 0.5f)
                        Soften(mission);
                    followCams[0] = Camera.main == null ? null : Camera.main.GetComponent("CargoAircraftCamera");
                    if (pickup != null && Env("FLOODUAV_FLEET", 0f) > 0.5f)
                        Fleet.Setup(mission, pickup);
                    if (Env("FLOODUAV_SPLIT", 0f) > 0.5f)
                        SplitViews(mission);
                    FollowOffset();
                }
                viewsReady = true;
                if (Fleet.Instance != null)
                    Fleet.Instance.ResetPickup();
                Layout.Apply(mission, pads);
                string path = Environment.GetEnvironmentVariable("FLOODUAV_INFO");
                if (!string.IsNullOrEmpty(path))
                    File.WriteAllText(path, Info(mission));
            }
            catch (Exception e)
            {
                Debug.LogError("[FloodPatch] " + e);
            }
        }

        static MonoBehaviour FindPickup(MonoBehaviour mission)
        {
            Type t = mission.GetType().Assembly.GetType("MBaske.SurvivorPickupAgent");
            UnityEngine.Object[] all = t == null ? new UnityEngine.Object[0] : UnityEngine.Object.FindObjectsOfType(t);
            return all.Length > 0 ? (MonoBehaviour)all[0] : null;
        }

        static void Soften(MonoBehaviour mission)
        {
            Component aircraft = (Component)Get(mission, "aircraft");
            Component failure = aircraft.GetComponent("DroneMissionFailure");
            if (failure != null)
                failure.GetType().GetField("failureLayers", F).SetValue(failure, (LayerMask)0);
            if (pickup != null)
                pickup.GetType().GetField("failureLayers", F).SetValue(pickup, (LayerMask)0);
            Debug.Log("[FloodPatch] contact failures off");
        }

        static void FollowOffset()
        {
            string s = Environment.GetEnvironmentVariable("FLOODUAV_FOLLOW");
            if (string.IsNullOrEmpty(s))
                return;
            string[] p = s.Split(',');
            Vector3 v = new Vector3(float.Parse(p[0], CultureInfo.InvariantCulture),
                float.Parse(p[1], CultureInfo.InvariantCulture), float.Parse(p[2], CultureInfo.InvariantCulture));
            for (int k = 0; k < followCams.Length; k++)
                FollowView(k, v);
        }

        // Offset of follow camera k from its drone.
        public static void FollowView(int k, Vector3 offset)
        {
            if (k >= 0 && k < followCams.Length && followCams[k] != null)
                followCams[k].GetType().GetField("offset").SetValue(followCams[k], offset);
        }

        // A copy of the main camera, its follow script and post-processing included, that follows `target`.
        static Component FollowCopy(Camera main, Transform target, float[] rect)
        {
            GameObject go = UnityEngine.Object.Instantiate(main.gameObject);
            go.name = "FloodPatchFollow1";
            go.tag = "Untagged";      // Camera.main stays the original
            Component ear = go.GetComponent("AudioListener");
            if (ear != null)
                UnityEngine.Object.Destroy(ear);
            go.GetComponent<Camera>().rect = R(rect);
            Component follow = go.GetComponent("CargoAircraftCamera");
            if (follow != null)
                follow.GetType().GetField("target").SetValue(follow, target);
            return follow;
        }

        static Camera DroneCam(Camera main, Transform parent, Quaternion local, float fov, float[] rect, float depth, float near)
        {
            GameObject go = new GameObject("FloodPatchDroneCam");
            go.transform.SetParent(parent, false);
            Camera cam = go.AddComponent<Camera>();
            cam.CopyFrom(main);       // copies the main camera's pose too
            go.transform.localPosition = Vector3.zero;
            go.transform.localRotation = local;
            cam.fieldOfView = fov;
            cam.nearClipPlane = near;
            cam.farClipPlane = Mathf.Max(main.farClipPlane, 2000f);
            cam.targetTexture = Capture.Target;
            cam.rect = R(rect);
            cam.depth = depth;
            cam.enabled = true;
            return cam;
        }

        static void SplitViews(MonoBehaviour mission)
        {
            object sensor = Get(mission, "visibilitySensor");
            Camera search = sensor == null ? null : Get(sensor, "searchCamera") as Camera;
            BoxCollider box = (BoxCollider)Get(mission, "searchBoundary");
            Camera main = Camera.main;
            RenderTexture target = Capture.Setup((int)SW, (int)SH);   // null: on screen
            foreach (Camera c in Camera.allCameras)
                if (c != search && c.targetTexture == null)
                {
                    c.rect = R(Follow);
                    c.targetTexture = target;
                }

            if (search != null && main != null)
            {
                // The sensor camera's own settings render nothing on screen; take the main
                // camera's and keep only the sensor camera's pose and field of view.
                Debug.Log(string.Format(CultureInfo.InvariantCulture,
                    "[FloodPatch] search camera fov={0} mask={1} near={2} far={3} enabled={4} forward={5}",
                    search.fieldOfView, search.cullingMask, search.nearClipPlane, search.farClipPlane, search.enabled,
                    V(search.transform.forward)));
                camBase[0] = Quaternion.identity;
                droneCams[0] = DroneCam(main, search.transform, camBase[0], search.fieldOfView, Cam0, 50f, 0.3f);
                // the pickup drone has no sensor camera: one straight down, top of the image forward,
                // from its centre, so the near plane clears its own hull
                if (pickup != null)
                {
                    camBase[1] = Quaternion.Euler(90f, 0f, 0f);
                    droneCams[1] = DroneCam(main, pickup.transform, camBase[1], search.fieldOfView, Cam1, 52f, 4f);
                }
            }
            if (pickup != null && main != null && followCams[0] != null)
                followCams[1] = FollowCopy(main, pickup.transform, Follow1);

            Bounds b = box.bounds;
            GameObject top = new GameObject("FloodPatchOverview");
            top.transform.position = new Vector3(b.center.x, b.max.y + 50f, b.center.z);
            top.transform.rotation = Quaternion.LookRotation(Vector3.down, box.transform.forward);
            Camera ov = top.AddComponent<Camera>();
            ov.orthographic = true;
            float aspect = Overview[2] / Overview[3];
            ov.orthographicSize = Mathf.Max(b.extents.z, b.extents.x / aspect);
            ov.nearClipPlane = 1f;
            ov.farClipPlane = b.size.y + 2000f;
            ov.rect = R(Overview);
            ov.targetTexture = Capture.Target;
            ov.depth = 51f;
            overview = ov;
            Debug.Log("[FloodPatch] split views ready");
        }

        // Zoom and gimbal of an on-screen drone camera; the sensor camera that finds survivors is not touched.
        // tilt: degrees from straight down toward the drone's forward; pan: degrees sideways.
        public static void DroneView(int k, float fov, float pan, float tilt)
        {
            if (k < 0 || k >= droneCams.Length || droneCams[k] == null)
                return;
            droneCams[k].fieldOfView = Mathf.Clamp(fov, 5f, 120f);
            droneCams[k].transform.localRotation = camBase[k] * Quaternion.Euler(-Mathf.Clamp(tilt, 0f, 89f), Mathf.Clamp(pan, -90f, 90f), 0f);
        }

        // One square top-down render of the overview camera at `size` pixels, written as a JPEG to FLOODUAV_MAP.
        public static void RenderMap(int size)
        {
            string path = Environment.GetEnvironmentVariable("FLOODUAV_MAP");
            if (overview == null || string.IsNullOrEmpty(path))
                return;
            size = Mathf.Clamp(size, 256, 8192);
            RenderTexture rt = new RenderTexture(size, size, 24);
            rt.antiAliasing = 4;
            Rect rect = overview.rect;
            Texture2D tex = new Texture2D(size, size, TextureFormat.RGB24, false);
            try
            {
                overview.targetTexture = rt;
                overview.rect = new Rect(0f, 0f, 1f, 1f);
                overview.Render();
                RenderTexture.active = rt;
                tex.ReadPixels(new Rect(0, 0, size, size), 0, 0);
                tex.Apply();
                File.WriteAllBytes(path + ".tmp", ImageConversion.EncodeToJPG(tex, 92));
                if (File.Exists(path))
                    File.Delete(path);
                File.Move(path + ".tmp", path);
                Debug.Log("[FloodPatch] map " + size + " px written to " + path);
            }
            finally
            {
                RenderTexture.active = null;
                overview.targetTexture = Capture.Target;
                overview.rect = rect;
                UnityEngine.Object.Destroy(rt);
                UnityEngine.Object.Destroy(tex);
            }
        }

        static string V(Vector3 v)
        {
            return string.Format(CultureInfo.InvariantCulture, "[{0:R},{1:R},{2:R}]", v.x, v.y, v.z);
        }

        static string Info(MonoBehaviour mission)
        {
            BoxCollider box = (BoxCollider)Get(mission, "searchBoundary");
            Transform start = (Transform)Get(mission, "droneStartPoint");
            Transform root = (Transform)Get(mission, "survivorsRoot");
            Transform t = box.transform;
            StringBuilder sb = new StringBuilder();
            sb.Append("{\"boundary\":{\"position\":").Append(V(t.position))
              .Append(",\"rotation\":").Append(string.Format(CultureInfo.InvariantCulture, "[{0:R},{1:R},{2:R},{3:R}]",
                  t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w))
              .Append(",\"scale\":").Append(V(t.lossyScale))
              .Append(",\"center\":").Append(V(box.center))
              .Append(",\"size\":").Append(V(box.size)).Append("}");
            sb.Append(",\"start\":").Append(V(t.InverseTransformPoint(start.position)));
            if (pickup != null)
                sb.Append(",\"pickup\":").Append(V(t.InverseTransformPoint(pickup.transform.position)));
            object sensor = Get(mission, "visibilitySensor");
            Camera search = sensor == null ? null : Get(sensor, "searchCamera") as Camera;
            sb.Append(",\"camera_fov\":").Append((search == null ? 60f : search.fieldOfView).ToString("R", CultureInfo.InvariantCulture));
            sb.Append(",\"cells\":").Append(Layout.CellsJson());
            sb.Append(",\"survivor_seed\":").Append(Layout.Seed.ToString(CultureInfo.InvariantCulture));
            sb.Append(",\"survivor_shares\":").Append(Layout.SharesJson());
            sb.Append(",\"survivors\":[");
            bool first = true;
            foreach (MonoBehaviour m in root.GetComponentsInChildren<MonoBehaviour>(true))
            {
                if (m.GetType().Name != "Survivor")
                    continue;
                PropertyInfo id = m.GetType().GetProperty("Id");
                PropertyInfo ok = m.GetType().GetProperty("IsSearchable");
                sb.Append(first ? "" : ",").Append("{\"id\":\"").Append(id.GetValue(m, null)).Append("\"")
                  .Append(",\"local\":").Append(V(t.InverseTransformPoint(m.transform.position)))
                  .Append(",\"kind\":\"").Append(Layout.Kind(m)).Append("\"")
                  .Append(",\"searchable\":").Append((bool)ok.GetValue(m, null) ? "true" : "false").Append("}");
                first = false;
            }
            sb.Append("]}");
            return sb.ToString();
        }
    }
}
