#if FLAX_EDITOR
using System.IO;
using FlaxEditor;
using FlaxEditor.Content;
using FlaxEngine;
using FlaxEngine.Tools;

/// <summary>
/// Imports Parallaxmapping.shader, displacement_plane.gltf, and texture assets into Content on editor load.
/// </summary>
public class ParallaxmappingEditorPlugin : EditorPlugin
{
    /// <inheritdoc />
    public override void InitializeEditor()
    {
        base.InitializeEditor();
        ImportShader();
        EnsureSourceAssets();
        ImportModels();
        ImportTextures();
    }

    private static void EnsureSourceAssets()
    {
        var texturesDir = Path.Combine(Globals.ProjectFolder, "Source", "Assets", "textures");
        Directory.CreateDirectory(texturesDir);

        CopyTextureIfMissing(texturesDir, "rocks_color_rgba.ktx", new[]
        {
            Path.Combine(Globals.ProjectFolder, "..", "..", "Vulkan", "assets", "textures", "rocks_color_rgba.ktx"),
        });

        CopyTextureIfMissing(texturesDir, "rocks_normal_height_rgba.ktx", new[]
        {
            Path.Combine(Globals.ProjectFolder, "..", "..", "Vulkan", "assets", "textures", "rocks_normal_height_rgba.ktx"),
        });

        var modelsDir = Path.Combine(Globals.ProjectFolder, "Source", "Assets", "models");
        Directory.CreateDirectory(modelsDir);
        CopyFileIfMissing(modelsDir, "plane.gltf", new[]
        {
            Path.Combine(Globals.ProjectFolder, "..", "..", "Vulkan", "assets", "models", "plane.gltf"),
        });
        CopyFileIfMissing(modelsDir, "displacement_plane.gltf", new[]
        {
            Path.Combine(Globals.ProjectFolder, "..", "..", "Vulkan", "assets", "models", "displacement_plane.gltf"),
        });
    }

    private static void CopyFileIfMissing(string destDir, string destFileName, string[] candidates)
    {
        var dest = Path.Combine(destDir, destFileName);
        if (File.Exists(dest))
            return;

        foreach (var candidate in candidates)
        {
            var full = Path.GetFullPath(candidate);
            if (!File.Exists(full))
                continue;

            File.Copy(full, dest, overwrite: false);
            Debug.Log($"parallaxmapping: copied {destFileName} from {full}");
            return;
        }

        Debug.LogWarning($"parallaxmapping: {destFileName} not found in Vulkan assets.");
    }

    private static void CopyTextureIfMissing(string texturesDir, string destFileName, string[] candidates)
    {
        var dest = Path.Combine(texturesDir, destFileName);
        if (File.Exists(dest))
            return;

        foreach (var candidate in candidates)
        {
            var full = Path.GetFullPath(candidate);
            if (!File.Exists(full))
                continue;

            File.Copy(full, dest, overwrite: false);
            Debug.Log($"parallaxmapping: copied {destFileName} from {full}");
            return;
        }

        Debug.LogWarning($"parallaxmapping: {destFileName} not found; init Vulkan assets submodule (git submodule update --init assets).");
    }

    private static void ImportShader()
    {
        ImportSourceFile("Shaders", "Parallaxmapping.shader", "Parallaxmapping.flax");
    }

    private static void ImportModels()
    {
        var modelsDir = Path.Combine(Globals.ProjectFolder, "Source", "Assets", "models");
        if (!Directory.Exists(modelsDir))
            return;

        var contentModels = Path.Combine(Globals.ProjectContentFolder, "Models");
        Directory.CreateDirectory(contentModels);

        foreach (var file in Directory.GetFiles(modelsDir, "*.gltf", SearchOption.TopDirectoryOnly))
        {
            var name = Path.GetFileNameWithoutExtension(file);
            var dest = Path.Combine(contentModels, name + ".flax");
            if (File.Exists(dest) && File.GetLastWriteTimeUtc(file) <= File.GetLastWriteTimeUtc(dest))
                continue;

            var folder = Editor.Instance.ContentDatabase.Find(contentModels) as ContentFolder
                ?? Editor.Instance.ContentDatabase.Find(Globals.ProjectContentFolder) as ContentFolder;
            if (folder == null)
                continue;

            Editor.Instance.ContentImporting.Import(file, folder, skipSettingsDialog: true);
            Debug.Log($"parallaxmapping: imported {name}.gltf");
        }
    }

    private static void ImportTextures()
    {
        var texturesDir = Path.Combine(Globals.ProjectFolder, "Source", "Assets", "textures");
        if (!Directory.Exists(texturesDir))
            return;

        var contentTextures = Path.Combine(Globals.ProjectContentFolder, "Textures");
        Directory.CreateDirectory(contentTextures);

        foreach (var file in Directory.GetFiles(texturesDir, "*.*", SearchOption.TopDirectoryOnly))
        {
            var ext = Path.GetExtension(file).ToLowerInvariant();
            if (ext != ".ktx" && ext != ".png" && ext != ".jpg" && ext != ".jpeg")
                continue;

            var name = Path.GetFileNameWithoutExtension(file);
            var dest = Path.Combine(contentTextures, name + ".flax");
            if (File.Exists(dest) && File.GetLastWriteTimeUtc(file) <= File.GetLastWriteTimeUtc(dest))
                continue;

            if (ext == ".ktx")
            {
                // Vulkan parallaxmapping.cpp loads both as VK_FORMAT_R8G8B8A8_UNORM (no sRGB decode).
                var importSettings = new TextureTool.Options
                {
                    GenerateMipMaps = false,
                    sRGB = false,
                    MaxSize = 8192,
                    Resize = false,
                };

                if (!Editor.Import(file, dest, importSettings))
                {
                    Debug.LogWarning($"parallaxmapping: failed to import {name}{ext}.");
                    continue;
                }
            }
            else
            {
                var folder = Editor.Instance.ContentDatabase.Find(contentTextures) as ContentFolder
                    ?? Editor.Instance.ContentDatabase.Find(Globals.ProjectContentFolder) as ContentFolder;
                if (folder == null)
                    continue;

                Editor.Instance.ContentImporting.Import(file, folder, skipSettingsDialog: true);
            }

            Debug.Log($"parallaxmapping: imported texture {name}");
        }
    }

    private static void ImportSourceFile(string contentSubdir, string sourceName, string destName)
    {
        var sourcePath = Path.Combine(Globals.ProjectFolder, "Source", contentSubdir.Replace('/', Path.DirectorySeparatorChar), sourceName);
        if (!File.Exists(sourcePath))
            return;

        var outputDir = Path.Combine(Globals.ProjectContentFolder, contentSubdir);
        var outputPath = Path.Combine(outputDir, destName);
        Directory.CreateDirectory(outputDir);

        if (File.Exists(outputPath) && File.GetLastWriteTimeUtc(sourcePath) <= File.GetLastWriteTimeUtc(outputPath))
            return;

        var folder = Editor.Instance.ContentDatabase.Find(outputDir) as ContentFolder
            ?? Editor.Instance.ContentDatabase.Find(Globals.ProjectContentFolder) as ContentFolder;
        if (folder == null)
            return;

        Editor.Instance.ContentImporting.Import(sourcePath, folder, skipSettingsDialog: true);
        Debug.Log($"parallaxmapping: imported {sourceName}");
    }
}
#endif
