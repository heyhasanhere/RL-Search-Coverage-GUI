// The build's pickup drone as a second mission drone (sim/flooduav/README.md).
// Its own pilot and pickup coordinator are switched off; the bridge flies it through
// an ML-Agents side channel, like the search drone through its agent actions.
//
// Bridge to Unity (float lists):
//   [1, forward, sideways, vertical, yaw]   pickup drone commands, -1..1 (DroneVelocityController)
//   [2, drone]                              follow camera target: 0 search drone, 1 pickup drone
//   [3]                                     pickup drone back to its base pad
// Unity to bridge, every decision step (5 physics steps):
//   [pos x, y, z normalized in the boundary, velocity x, y, z in the boundary frame (m/s),
//    heading sin, cos (from boundary forward, clockwise), nearest of 8 horizontal rays / 30 m,
//    down ray hit flag, down ray / 150 m]

using System;
using System.Collections.Generic;
using System.Reflection;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using UnityEngine;

namespace FloodPatch
{
    public sealed class FleetChannel : SideChannel
    {
        public FleetChannel()
        {
            ChannelId = new Guid("6f1c2b6e-5a0d-4e8b-9a43-0d2f4c7b9e11");
        }

        protected override void OnMessageReceived(IncomingMessage msg)
        {
            if (Fleet.Instance != null)
                Fleet.Instance.Receive(msg.ReadFloatList());
        }

        public void Send(List<float> values)
        {
            using (OutgoingMessage m = new OutgoingMessage())
            {
                m.WriteFloatList(values);
                QueueMessageToSend(m);
            }
        }
    }

    public sealed class Fleet : MonoBehaviour
    {
        const BindingFlags F = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public;
        public static Fleet Instance;

        FleetChannel channel;
        MonoBehaviour agent, controller;
        MethodInfo setCommands;
        Rigidbody body;
        Transform pickupBase;
        BoxCollider boundary;
        LayerMask mask;
        Component follow;
        Transform[] targets;
        readonly object[] cmd = { 0f, 0f, 0f, 0f };

        public static void Setup(MonoBehaviour mission, MonoBehaviour pickupAgent)
        {
            GameObject go = new GameObject("FloodPatchFleet");
            Fleet f = go.AddComponent<Fleet>();
            Instance = f;
            f.agent = pickupAgent;
            f.boundary = (BoxCollider)Hook.Get(mission, "searchBoundary");
            f.body = pickupAgent.GetComponent<Rigidbody>();
            f.controller = (MonoBehaviour)pickupAgent.GetComponent("DroneVelocityController");
            f.setCommands = f.controller.GetType().GetMethod("SetNormalizedCommands");
            f.mask = (LayerMask)Hook.Get(pickupAgent, "environmentMask");

            // The pickup pilot and coordinator would fly the drone and end episodes.
            MonoBehaviour pilot = (MonoBehaviour)pickupAgent.GetComponent("PickupAStarPilot");
            if (pilot != null)
                pilot.enabled = false;
            MonoBehaviour coord = (MonoBehaviour)Hook.Get(pickupAgent, "coordinator");
            f.pickupBase = (Transform)coord.GetType().GetProperty("Base").GetValue(coord, null);
            coord.GetType().GetField("cycleActive", F).SetValue(coord, false);
            coord.enabled = false;
            pickupAgent.GetType().GetMethod("SetExternalNavigation").Invoke(pickupAgent, new object[] { true });

            Component aircraft = (Component)Hook.Get(mission, "aircraft");
            f.targets = new[] { aircraft.transform, pickupAgent.transform };
            f.follow = Camera.main == null ? null : Camera.main.GetComponent("CargoAircraftCamera");

            f.channel = new FleetChannel();
            SideChannelManager.RegisterSideChannel(f.channel);
            Academy.Instance.AgentPreStep += f.PreStep;
            Debug.Log("[FloodPatch] fleet ready: pickup drone " + pickupAgent.name);
        }

        public void ResetPickup()
        {
            for (int i = 0; i < 4; i++)
                cmd[i] = 0f;
            agent.GetType().GetMethod("ResetFlightAtBase").Invoke(agent, new object[] { pickupBase });
        }

        public void Receive(IList<float> v)
        {
            if (v == null || v.Count == 0)
                return;
            int kind = Mathf.RoundToInt(v[0]);
            if (kind == 1 && v.Count >= 5)
                for (int i = 0; i < 4; i++)
                    cmd[i] = Mathf.Clamp(v[i + 1], -1f, 1f);
            else if (kind == 2 && v.Count >= 2 && follow != null)
            {
                int k = Mathf.Clamp(Mathf.RoundToInt(v[1]), 0, targets.Length - 1);
                follow.GetType().GetField("target").SetValue(follow, targets[k]);
            }
            else if (kind == 3)
                ResetPickup();
        }

        void FixedUpdate()
        {
            if (controller != null)
                setCommands.Invoke(controller, cmd);
        }

        void PreStep(int step)
        {
            if (step % 5 != 0 || body == null)
                return;
            Transform t = boundary.transform;
            Vector3 half = boundary.size * 0.5f;
            Vector3 p = t.InverseTransformPoint(body.position) - boundary.center;
            Vector3 v = t.InverseTransformDirection(body.linearVelocity);
            Vector3 fwd = Vector3.ProjectOnPlane(body.rotation * Vector3.forward, Vector3.up);
            if (fwd.sqrMagnitude < 0.001f)
                fwd = Vector3.forward;
            fwd.Normalize();
            Vector3 h = t.InverseTransformDirection(fwd);
            float hn = Mathf.Max(new Vector2(h.x, h.z).magnitude, 1e-4f);
            float ray = 1f;
            RaycastHit hit;
            for (int i = 0; i < 8; i++)
                if (Physics.Raycast(body.position, Quaternion.AngleAxis(i * 45f, Vector3.up) * fwd, out hit, 30f, mask, QueryTriggerInteraction.Collide))
                    ray = Mathf.Min(ray, hit.distance / 30f);
            bool down = Physics.Raycast(body.position, Vector3.down, out hit, 150f, mask, QueryTriggerInteraction.Collide);
            channel.Send(new List<float> {
                p.x / half.x, p.y / half.y, p.z / half.z, v.x, v.y, v.z, h.x / hn, h.z / hn, ray,
                down ? 1f : 0f, down ? hit.distance / 150f : 1f });
        }
    }
}
