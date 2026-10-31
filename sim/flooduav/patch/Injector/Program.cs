// Adds calls to FloodPatch.Hook into the FloodUAV game code.
// Usage: Injector <original Assembly-CSharp.dll> <FloodPatch.dll> <output Assembly-CSharp.dll>

using System;
using System.IO;
using System.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;

var (src, hookPath, dst) = (args[0], args[1], args[2]);
var resolver = new DefaultAssemblyResolver();
resolver.AddSearchDirectory(Path.GetDirectoryName(Path.GetFullPath(src))!);
var asm = AssemblyDefinition.ReadAssembly(src, new ReaderParameters { AssemblyResolver = resolver });
var hooks = AssemblyDefinition.ReadAssembly(hookPath, new ReaderParameters { AssemblyResolver = resolver })
    .MainModule.GetType("FloodPatch.Hook");
var mod = asm.MainModule;

void Prefix(string type, string method, string hook)
{
    var m = mod.GetType(type).Methods.Single(x => x.Name == method && !x.HasParameters);
    var call = mod.ImportReference(hooks.Methods.Single(x => x.Name == hook));
    var il = m.Body.GetILProcessor();
    var first = m.Body.Instructions[0];
    il.InsertBefore(first, il.Create(OpCodes.Ldarg_0));
    il.InsertBefore(first, il.Create(OpCodes.Call, call));
    Console.WriteLine($"patched {type}.{method} -> Hook.{hook}");
}

// Like Prefix, but the method returns right away when the hook returns true.
void Skip(string type, string method, string hook)
{
    var m = mod.GetType(type).Methods.Single(x => x.Name == method && !x.HasParameters);
    var call = mod.ImportReference(hooks.Methods.Single(x => x.Name == hook));
    var il = m.Body.GetILProcessor();
    var first = m.Body.Instructions[0];
    il.InsertBefore(first, il.Create(OpCodes.Ldarg_0));
    il.InsertBefore(first, il.Create(OpCodes.Call, call));
    il.InsertBefore(first, il.Create(OpCodes.Brfalse, first));
    il.InsertBefore(first, il.Create(OpCodes.Ret));
    Console.WriteLine($"patched {type}.{method} -> Hook.{hook} (skip)");
}

Prefix("MBaske.DroneVelocityController", "OnEnable", "Drone");
Prefix("MBaske.SearchMissionManager", "BeginMission", "Mission");
Skip("MBaske.MissionMapDisplay", "Start", "MapDisplay");
asm.Write(dst);
