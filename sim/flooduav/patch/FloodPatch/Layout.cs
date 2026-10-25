// Survivor positions and scene cell types (sim/flooduav/README.md).
// Seed < 0 keeps the build's fixed survivors. Any other seed scatters them by the flood
// survivor shares of sar/prior/flood_bank.py (nobody inside a building):
//   roof         50%  tops of flooded buildings above the water
//   high ground  25%  dry ground near the flood edge, weight exp(-d^2 / 2 sigma^2), dry islands doubled
//   bridge       10%  bridge decks above the water
//   water        10%  in open flood water, upper body above the surface
//   random        5%  any outdoor surface or water in the area
// The same seed gives the same positions; the layout is made once per seed and put back at
// every mission start, so a reset keeps it.
// Count: how many survivors are in the town. Fewer than the build's are hidden; more are copies
// of its first survivor (ids Survivor_016 and on), which need a random layout to have a place.
//
// Cell types: the area is sampled every 10 m once, and each sample gets one character (CellTypes):
//   .  dry ground      w  flood water        r  flooded building, roof above the water
//   s  submerged building   b  bridge deck above the water   h  dry building   o  nothing hit
// The info file carries them, and the mission server builds the flood prior from them.
//
// Environment variables (optional):
//   FLOODUAV_SURVIVOR_SEED   seed of the first layout (default -1: the build's survivors)
//   FLOODUAV_SURVIVOR_COUNT  number of survivors (default -1: the build's 15)

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Reflection;
using UnityEngine;

namespace FloodPatch
{
    public static class Layout
    {
        // sar/prior/flood_bank.py SHARES, in the order of Kinds
        static readonly string[] Kinds = { "roof", "high_ground", "bridge", "water", "random" };
        static readonly float[] Shares = { 0.50f, 0.25f, 0.10f, 0.10f, 0.05f };
        static readonly char[] KindCells = { 'r', '.', 'b', 'w', '*' };
        const float SampleStep = 10f;      // cell type sample spacing, m
        const float WaterDepthShown = 0.6f; // a person in the water stands this far below its surface
        const float Freeboard = 0.5f;      // ROOF_FREEBOARD_M: a standing surface this far above the water
        const float HighSigma = 90f;       // HIGH_SIGMA_CELLS x 30 m: how far people walk or wade to dry ground
        const float HighReach = 240f;      // HIGH_REACH_CELLS x 30 m: dry ground further from the flood gets nothing
        const float IslandBonus = 2f;      // ISLAND_BONUS: dry islands hold people who cannot walk out
        const float MinStructure = 2.5f;   // structure top above the ground below it
        const float MinSlopeCos = 0.75f;   // steepest surface a person stands on (about 40 degrees)
        const float Spacing = 10f;         // between survivors; more than the mission's 8 m merge distance
        const float PadClear = 25f;        // kept clear around the drone pads
        const float Margin = 0.05f;        // fraction of the boundary kept clear on each side
        const float Step = 10f;            // flood search step along each of 8 directions
        const int Tries = 20000;           // candidate points per survivor
        const int MaxCount = 100;

        struct Pose
        {
            public MonoBehaviour survivor;
            public Vector3 position;
            public Quaternion rotation;
            public string kind;
        }

        static bool ready;
        static int seed = -1, count = -1;
        static Pose[] poses;
        static readonly List<Pose> bases = new List<Pose>();   // the build's survivors at their own places, then copies
        static Transform root;
        static readonly List<Bounds> water = new List<Bounds>();
        static int topMask, groundMask;
        static Vector3 sampleOrigin;       // boundary-local position of the first sample
        static float sampleDx, sampleDz;   // boundary-local step between samples
        static int sampleNx, sampleNz;
        static char[] cellTypes;

        public static int Seed { get { return seed; } }

        // A new layout of `n` survivors (-1: the build's number); it takes effect at the next mission start.
        public static void Set(int value, int n)
        {
            seed = value;
            count = n;
            poses = null;
        }

        public static string Kind(MonoBehaviour survivor)
        {
            if (poses != null)
                foreach (Pose p in poses)
                    if (p.survivor == survivor)
                        return p.kind;
            return "fixed";
        }

        // Called at the start of every mission, before Unity registers the survivors.
        public static void Apply(MonoBehaviour mission, Vector3[] pads)
        {
            if (!ready)
            {
                Init(mission);
                SampleCells(mission);
                ready = true;
            }
            if (poses == null)
            {
                int fixedCount = bases.FindAll(b => b.kind == "fixed").Count;
                int n = count < 0 ? fixedCount : Mathf.Clamp(count, 1, seed < 0 ? fixedCount : MaxCount);
                while (bases.Count < n)
                    bases.Add(Copy(bases.Count + 1));
                poses = seed < 0 ? bases.GetRange(0, n).ToArray() : Make(mission, pads, seed, n);
                for (int i = 0; i < bases.Count; i++)
                {
                    MonoBehaviour m = bases[i].survivor;
                    m.gameObject.SetActive(i < n);
                    m.GetType().GetMethod("SetAvailableForSearch").Invoke(m, new object[] { i < n });
                }
            }
            foreach (Pose p in poses)
                p.survivor.transform.SetPositionAndRotation(p.position, p.rotation);
        }

        // A copy of the build's first survivor with the next id; Make places it.
        static Pose Copy(int number)
        {
            Pose first = bases[0];
            GameObject go = UnityEngine.Object.Instantiate(first.survivor.gameObject, root);
            string id = "Survivor_" + number.ToString("000", CultureInfo.InvariantCulture);
            go.name = id;
            MonoBehaviour m = (MonoBehaviour)go.GetComponent(first.survivor.GetType());
            m.GetType().GetField("survivorId", BindingFlags.Instance | BindingFlags.NonPublic).SetValue(m, id);
            return new Pose { survivor = m, position = first.position, rotation = first.rotation, kind = "copy" };
        }

        static void Init(MonoBehaviour mission)
        {
            string s = Environment.GetEnvironmentVariable("FLOODUAV_SURVIVOR_SEED");
            int v;
            if (s != null && int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out v))
                seed = v;
            s = Environment.GetEnvironmentVariable("FLOODUAV_SURVIVOR_COUNT");
            if (s != null && int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out v))
                count = v;
            topMask = LayerMask.GetMask("Default", "Obstacle", "Ground");
            groundMask = LayerMask.GetMask("Ground");
            root = (Transform)Hook.Get(mission, "survivorsRoot");
            List<Pose> list = new List<Pose>();
            foreach (MonoBehaviour m in root.GetComponentsInChildren<MonoBehaviour>(true))
                if (m.GetType().Name == "Survivor")
                    list.Add(new Pose { survivor = m, position = m.transform.position, rotation = m.transform.rotation, kind = "fixed" });
            PropertyInfo id = list.Count > 0 ? list[0].survivor.GetType().GetProperty("Id") : null;
            list.Sort((a, b) => string.CompareOrdinal((string)id.GetValue(a.survivor, null), (string)id.GetValue(b.survivor, null)));
            bases.AddRange(list);
            // The flood is drawn by water meshes without colliders; their tops are the water surface.
            foreach (Renderer r in UnityEngine.Object.FindObjectsByType<Renderer>(FindObjectsSortMode.None))
                if (IsWater(r.transform))
                    water.Add(r.bounds);
            Debug.Log("[FloodPatch] survivors " + bases.Count + ", water meshes " + water.Count);
        }

        static bool IsWater(Transform t)
        {
            if (t.gameObject.layer == LayerMask.NameToLayer("Water"))
                return true;
            for (; t != null; t = t.parent)
                if (t.name.StartsWith("3D_Water") || t.name.StartsWith("WaterPlane"))
                    return true;
            return false;
        }

        static float WaterAt(float x, float z)
        {
            float h = float.NegativeInfinity;
            foreach (Bounds b in water)
                if (x >= b.min.x && x <= b.max.x && z >= b.min.z && z <= b.max.z)
                    h = Mathf.Max(h, b.max.y);
            return h;
        }

        static bool Flooded(float x, float z)
        {
            RaycastHit hit;
            return Physics.Raycast(new Vector3(x, 1000f, z), Vector3.down, out hit, 2000f, groundMask, QueryTriggerInteraction.Ignore)
                && hit.point.y < WaterAt(x, z);
        }

        // Distance to the nearest flooded point along 8 directions, up to `reach`; island: flood in all 8.
        static float FloodDistance(Vector3 p, float reach, out bool island)
        {
            float best = float.PositiveInfinity;
            int closed = 0;
            for (int i = 0; i < 8; i++)
            {
                float a = i * Mathf.PI / 4f;
                for (float r = Step; r <= reach; r += Step)
                    if (Flooded(p.x + r * Mathf.Sin(a), p.z + r * Mathf.Cos(a)))
                    {
                        best = Mathf.Min(best, r);
                        closed++;
                        break;
                    }
            }
            island = closed == 8;
            return best;
        }

        // Flat enough to stand on: the surface 0.7 m around is within 0.4 m of the point.
        static bool Flat(Vector3 p)
        {
            RaycastHit hit;
            for (int i = 0; i < 4; i++)
            {
                Vector3 q = p + Quaternion.Euler(0f, i * 90f, 0f) * new Vector3(0.7f, 0f, 0f);
                if (!Physics.Raycast(q + Vector3.up * 3f, Vector3.down, out hit, 6f, topMask, QueryTriggerInteraction.Ignore)
                    || Mathf.Abs(hit.point.y - p.y) > 0.4f)
                    return false;
            }
            return true;
        }

        static bool IsBridge(Transform t)
        {
            for (; t != null; t = t.parent)
                if (t.name.IndexOf("bridge", StringComparison.OrdinalIgnoreCase) >= 0)
                    return true;
            return false;
        }

        // Cell type of the outdoor surface at (x, z) world; `at` is that surface (or the water surface).
        static char Classify(float x, float z, float top, out Vector3 at)
        {
            RaycastHit hit, g;
            at = new Vector3(x, 0f, z);
            if (!Physics.Raycast(new Vector3(x, top, z), Vector3.down, out hit, 2000f, topMask, QueryTriggerInteraction.Ignore))
                return 'o';
            at = hit.point;
            float wl = WaterAt(x, z);
            bool hasGround = Physics.Raycast(new Vector3(x, top, z), Vector3.down, out g, 2000f, groundMask, QueryTriggerInteraction.Ignore);
            float ground = hasGround ? g.point.y : hit.point.y;
            bool onGround = hit.collider.gameObject.layer == LayerMask.NameToLayer("Ground");
            if (onGround || hit.point.y - ground < MinStructure)
            {
                if (hit.point.y < wl)
                {
                    at.y = wl;
                    return 'w';
                }
                return '.';
            }
            bool above = hit.point.y >= wl + Freeboard;
            bool flooded = ground < wl;
            // walls, fences and pitched edges: nobody stands there, the cell below counts
            bool standable = hit.normal.y >= MinSlopeCos && Flat(hit.point);
            if (IsBridge(hit.collider.transform))
            {
                if (!flooded)
                    return '.';           // an elevated road over dry land is a road
                if (above && standable)
                    return 'b';
                at.y = wl;
                return 'w';
            }
            if (flooded)
            {
                if (!above)
                    return 's';
                if (standable)
                    return 'r';
                at.y = wl;
                return 'w';
            }
            return 'h';
        }

        static void SampleCells(MonoBehaviour mission)
        {
            if (cellTypes != null)
                return;
            float t0 = Time.realtimeSinceStartup;
            BoxCollider box = (BoxCollider)Hook.Get(mission, "searchBoundary");
            Transform bt = box.transform;
            Vector3 sc = bt.lossyScale;
            sampleDx = SampleStep / Mathf.Abs(sc.x);
            sampleDz = SampleStep / Mathf.Abs(sc.z);
            sampleNx = Mathf.Max(1, Mathf.FloorToInt(box.size.x / sampleDx));
            sampleNz = Mathf.Max(1, Mathf.FloorToInt(box.size.z / sampleDz));
            sampleOrigin = box.center - new Vector3((sampleNx - 1) * sampleDx, 0f, (sampleNz - 1) * sampleDz) * 0.5f;
            float top = box.bounds.max.y + 10f;
            cellTypes = new char[sampleNx * sampleNz];
            int[] counts = new int[128];
            for (int j = 0; j < sampleNz; j++)
                for (int i = 0; i < sampleNx; i++)
                {
                    Vector3 w = bt.TransformPoint(sampleOrigin + new Vector3(i * sampleDx, 0f, j * sampleDz));
                    Vector3 at;
                    char c = Classify(w.x, w.z, top, out at);
                    cellTypes[j * sampleNx + i] = c;
                    counts[c]++;
                }
            Debug.Log(string.Format(CultureInfo.InvariantCulture,
                "[FloodPatch] cell types {0}x{1}: dry {2}, water {3}, roof {4}, submerged {5}, bridge {6}, dry building {7}, none {8}, {9} ms",
                sampleNx, sampleNz, counts['.'], counts['w'], counts['r'], counts['s'], counts['b'], counts['h'], counts['o'],
                Mathf.RoundToInt((Time.realtimeSinceStartup - t0) * 1000f)));
        }

        // JSON object of the cell type samples, boundary-local; empty before the first mission start.
        public static string CellsJson()
        {
            if (cellTypes == null)
                return "null";
            return string.Format(CultureInfo.InvariantCulture,
                "{{\"origin\":[{0:R},{1:R},{2:R}],\"dx\":{3:R},\"dz\":{4:R},\"nx\":{5},\"nz\":{6},\"types\":\"{7}\"}}",
                sampleOrigin.x, sampleOrigin.y, sampleOrigin.z, sampleDx, sampleDz, sampleNx, sampleNz, new string(cellTypes));
        }

        static int DrawKind(System.Random rng)
        {
            double u = rng.NextDouble(), acc = 0;
            for (int k = 0; k < Shares.Length; k++)
            {
                acc += Shares[k];
                if (u < acc)
                    return k;
            }
            return Shares.Length - 1;
        }

        static Pose[] Make(MonoBehaviour mission, Vector3[] pads, int s, int n)
        {
            float t0 = Time.realtimeSinceStartup;
            SampleCells(mission);
            System.Random rng = new System.Random(s);
            BoxCollider box = (BoxCollider)Hook.Get(mission, "searchBoundary");
            Transform bt = box.transform;
            float top = box.bounds.max.y + 10f;
            int mx = Mathf.CeilToInt(sampleNx * Margin), mz = Mathf.CeilToInt(sampleNz * Margin);
            // candidate samples per kind, inside the margin
            List<int>[] cand = new List<int>[Kinds.Length];
            for (int k = 0; k < Kinds.Length; k++)
                cand[k] = new List<int>();
            for (int j = mz; j < sampleNz - mz; j++)
                for (int i = mx; i < sampleNx - mx; i++)
                {
                    int idx = j * sampleNx + i;
                    char c = cellTypes[idx];
                    for (int k = 0; k < Kinds.Length; k++)
                        if (KindCells[k] == c || (KindCells[k] == '*' && c != 'o' && c != 's'))
                            cand[k].Add(idx);
                }
            Pose[] result = new Pose[n];
            List<Vector3> placed = new List<Vector3>();
            int[] got = new int[Kinds.Length];
            int kept = 0;
            for (int k = 0; k < result.Length; k++)
            {
                Pose p = bases[k];
                int kind = DrawKind(rng);
                for (int tries = 0; cand[kind].Count == 0 && tries < 10; tries++)
                    kind = DrawKind(rng);          // a scene without bridges gives their share to the others
                bool found = false;
                for (int t = 0; t < Tries && !found && cand[kind].Count > 0; t++)
                {
                    int idx = cand[kind][rng.Next(cand[kind].Count)];
                    double u = rng.NextDouble();      // drawn every try, so one rejection does not shift the rest
                    float ox = (float)(rng.NextDouble() - 0.5) * sampleDx, oz = (float)(rng.NextDouble() - 0.5) * sampleDz;
                    Vector3 local = sampleOrigin + new Vector3((idx % sampleNx) * sampleDx + ox, 0f, (idx / sampleNx) * sampleDz + oz);
                    Vector3 w = bt.TransformPoint(local);
                    Vector3 at;
                    char c = Classify(w.x, w.z, top, out at);
                    if (c == 'o' || c == 's' || (KindCells[kind] != '*' && c != KindCells[kind]))
                        continue;
                    if (Near(at, pads, PadClear) || Near(at, placed, Spacing))
                        continue;
                    if (c != 'w' && !Flat(at))
                        continue;
                    if (c == 'w')
                        at.y -= WaterDepthShown;
                    if (Kinds[kind] == "high_ground")
                    {
                        bool island;
                        float d = FloodDistance(at, HighReach, out island);
                        float weight = d > HighReach ? 0f : Mathf.Exp(-d * d / (2f * HighSigma * HighSigma)) * (island ? IslandBonus : 1f);
                        if (u * IslandBonus >= weight)
                            continue;
                    }
                    float yaw = (float)(rng.NextDouble() * 360.0);
                    Vector3 e = p.rotation.eulerAngles;
                    result[k] = new Pose { survivor = p.survivor, position = at, rotation = Quaternion.Euler(e.x, yaw, e.z), kind = Kinds[kind] };
                    placed.Add(at);
                    found = true;
                }
                if (found)
                    got[kind]++;
                else
                {
                    result[k] = p;            // nowhere found: the build's own position (a copy: the first survivor's)
                    kept++;
                }
            }
            Debug.Log(string.Format(CultureInfo.InvariantCulture,
                "[FloodPatch] survivor layout seed {0}, {1} survivors: {2} roof, {3} high ground, {4} bridge, {5} water, {6} random, {7} not placed, {8} ms",
                s, n, got[0], got[1], got[2], got[3], got[4], kept, Mathf.RoundToInt((Time.realtimeSinceStartup - t0) * 1000f)));
            return result;
        }

        static bool Near(Vector3 p, IEnumerable<Vector3> others, float d)
        {
            foreach (Vector3 o in others)
                if (new Vector2(p.x - o.x, p.z - o.z).sqrMagnitude < d * d)
                    return true;
            return false;
        }
    }
}
