#if FLAX_EDITOR
using System.IO;
using FlaxEditor;
using FlaxEditor.Content;
using FlaxEngine;
using FlaxEngine.Tools;

/// <summary>
/// Imports Texture.shader and metal plate KTX into Content on editor load.
/// </summary>
public class TextureEditorPlugin : EditorPlugin
{
    /// <inheritdoc />
    public override void InitializeEditor()
    {
        base.InitializeEditor();
        EnsureShaderAsset();
        EnsureMetalPlateTextureAsset();
    }

    private static void EnsureShaderAsset()
    {
        var sourcePath = Path.Combine(Globals.ProjectFolder, "Source", "Shaders", "Texture.shader");
        if (!File.Exists(sourcePath))
            return;

        var outputDir = Path.Combine(Globals.ProjectContentFolder, "Shaders");
        var outputPath = Path.Combine(outputDir, "Texture.flax");
        Directory.CreateDirectory(outputDir);

        var needsImport = !File.Exists(outputPath)
            || File.GetLastWriteTimeUtc(sourcePath) > File.GetLastWriteTimeUtc(outputPath);

        if (!needsImport)
            return;

        var targetFolder = Editor.Instance.ContentDatabase.Find(outputDir) as ContentFolder
            ?? Editor.Instance.ContentDatabase.Find(Globals.ProjectContentFolder) as ContentFolder;
        if (targetFolder == null)
        {
            Debug.LogWarning("texture: Content folder not found; cannot import Texture.shader.");
            return;
        }

        Editor.Instance.ContentImporting.Import(sourcePath, targetFolder, skipSettingsDialog: true);
        Debug.Log("texture: importing Texture.shader to Content/Shaders/Texture.flax");
    }

    private static void EnsureMetalPlateTextureAsset()
    {
        var ktxPath = Path.Combine(Globals.ProjectFolder, "Source", "Assets", "textures", "metalplate01_rgba.ktx");
        if (!File.Exists(ktxPath))
            return;

        var outputDir = Path.Combine(Globals.ProjectContentFolder, "Textures");
        var outputPath = Path.Combine(outputDir, "metalplate01_rgba.flax");
        Directory.CreateDirectory(outputDir);

        var needsImport = !File.Exists(outputPath)
            || File.GetLastWriteTimeUtc(ktxPath) > File.GetLastWriteTimeUtc(outputPath);

        if (!needsImport)
            return;

        var importSettings = new TextureTool.Options
        {
            GenerateMipMaps = false,
            sRGB = true,
            MaxSize = 8192,
            Resize = false,
        };

        if (!Editor.Import(ktxPath, outputPath, importSettings))
        {
            Debug.LogWarning("texture: failed to import metalplate01_rgba.ktx.");
            return;
        }

        Debug.Log("texture: imported metalplate01_rgba.ktx to Content/Textures/metalplate01_rgba.flax");
    }
}
#endif
