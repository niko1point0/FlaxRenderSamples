#if FLAX_EDITOR
using System.IO;
using FlaxEditor;
using FlaxEditor.Content;
using FlaxEngine;

/// <summary>
/// Imports Shadowmapping.shader and glTF models into Content/Models on editor load.
/// </summary>
public class ShadowmappingEditorPlugin : EditorPlugin
{
    static readonly string VulkanAssets = Path.GetFullPath(
        Path.Combine(Globals.ProjectFolder, "..", "..", "..", "Vulkan", "assets"));

    static readonly string[] ModelNames = { "vulkanscene_shadow", "samplescene" };

    /// <inheritdoc />
    public override void InitializeEditor()
    {
        base.InitializeEditor();
        EnsureSourceModels();
        RelocateRootImports();
        ImportShader();
        ImportModels();
        RelocateRootImports();
    }

    private static void EnsureSourceModels()
    {
        var modelsDir = Path.Combine(Globals.ProjectFolder, "Source", "Assets", "models");
        Directory.CreateDirectory(modelsDir);

        foreach (var name in ModelNames)
        {
            var dest = Path.Combine(modelsDir, name + ".gltf");
            var source = Path.Combine(VulkanAssets, "models", name + ".gltf");
            if (!File.Exists(source))
            {
                Debug.LogWarning($"shadowmapping: missing Vulkan asset {name}.gltf");
                continue;
            }

            if (!File.Exists(dest) || File.GetLastWriteTimeUtc(source) > File.GetLastWriteTimeUtc(dest))
                File.Copy(source, dest, overwrite: true);
        }
    }

    private static void RelocateRootImports()
    {
        var contentModels = Path.Combine(Globals.ProjectContentFolder, "Models");
        Directory.CreateDirectory(contentModels);

        foreach (var name in ModelNames)
        {
            var rootFlax = Path.Combine(Globals.ProjectContentFolder, name + ".flax");
            var modelsFlax = Path.Combine(contentModels, name + ".flax");
            if (File.Exists(rootFlax) && !File.Exists(modelsFlax))
            {
                File.Move(rootFlax, modelsFlax);
                Debug.Log($"shadowmapping: moved {name}.flax to Content/Models");
            }

            var rootDir = Path.Combine(Globals.ProjectContentFolder, name);
            var modelsDir = Path.Combine(contentModels, name);
            if (Directory.Exists(rootDir) && !Directory.Exists(modelsDir))
            {
                Directory.Move(rootDir, modelsDir);
                Debug.Log($"shadowmapping: moved {name}/ materials to Content/Models/{name}");
            }
        }
    }

    private static ContentFolder GetImportFolder(string folderPath)
    {
        var folder = Editor.Instance.ContentDatabase.Find(folderPath) as ContentFolder;
        if (folder != null)
            return folder;

        var root = Editor.Instance.ContentDatabase.Find(Globals.ProjectContentFolder) as ContentFolder;
        if (root != null)
            Editor.Instance.ContentDatabase.RefreshFolder(root, true);

        return Editor.Instance.ContentDatabase.Find(folderPath) as ContentFolder
            ?? Editor.Instance.ContentDatabase.Find(Globals.ProjectContentFolder) as ContentFolder;
    }

    private static void ImportModels()
    {
        var modelsDir = Path.Combine(Globals.ProjectFolder, "Source", "Assets", "models");
        if (!Directory.Exists(modelsDir))
            return;

        var contentModels = Path.Combine(Globals.ProjectContentFolder, "Models");
        Directory.CreateDirectory(contentModels);

        var folder = GetImportFolder(contentModels);
        if (folder == null)
        {
            Debug.LogWarning("shadowmapping: Content folder not found; cannot import models.");
            return;
        }

        foreach (var name in ModelNames)
        {
            var source = Path.Combine(modelsDir, name + ".gltf");
            if (!File.Exists(source))
                continue;

            var dest = Path.Combine(contentModels, name + ".flax");
            if (File.Exists(dest) && File.GetLastWriteTimeUtc(source) <= File.GetLastWriteTimeUtc(dest))
                continue;

            Editor.Instance.ContentImporting.Import(source, folder, skipSettingsDialog: true);
            Debug.Log($"shadowmapping: importing {name}.gltf -> Content/Models/{name}.flax");
        }

        var modelsFolder = Editor.Instance.ContentDatabase.Find(contentModels) as ContentFolder;
        if (modelsFolder != null)
            Editor.Instance.ContentDatabase.RefreshFolder(modelsFolder, true);
    }

    private static void ImportShader()
    {
        var sourcePath = Path.Combine(Globals.ProjectFolder, "Source", "Shaders", "Shadowmapping.shader");
        if (!File.Exists(sourcePath))
            return;

        var outputDir = Path.Combine(Globals.ProjectContentFolder, "Shaders");
        var outputPath = Path.Combine(outputDir, "Shadowmapping.flax");
        Directory.CreateDirectory(outputDir);

        var folder = GetImportFolder(outputDir);
        if (folder == null)
            return;

        Editor.Instance.ContentImporting.Import(sourcePath, folder, skipSettingsDialog: true);
        Debug.Log("shadowmapping: imported Shadowmapping.shader");
    }
}
#endif
