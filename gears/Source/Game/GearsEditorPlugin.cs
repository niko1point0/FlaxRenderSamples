#if FLAX_EDITOR
using System.IO;
using FlaxEditor;
using FlaxEditor.Content;
using FlaxEngine;

/// <summary>
/// Imports Gears.shader into Content on editor load.
/// </summary>
public class GearsEditorPlugin : EditorPlugin
{
    /// <inheritdoc />
    public override void InitializeEditor()
    {
        base.InitializeEditor();
        EnsureShaderAsset();
    }

    private static void EnsureShaderAsset()
    {
        var sourcePath = Path.Combine(Globals.ProjectFolder, "Source", "Shaders", "Gears.shader");
        if (!File.Exists(sourcePath))
            return;

        var outputDir = Path.Combine(Globals.ProjectContentFolder, "Shaders");
        var outputPath = Path.Combine(outputDir, "Gears.flax");
        Directory.CreateDirectory(outputDir);

        var needsImport = !File.Exists(outputPath)
            || File.GetLastWriteTimeUtc(sourcePath) > File.GetLastWriteTimeUtc(outputPath);

        if (!needsImport)
            return;

        var targetFolder = Editor.Instance.ContentDatabase.Find(outputDir) as ContentFolder
            ?? Editor.Instance.ContentDatabase.Find(Globals.ProjectContentFolder) as ContentFolder;
        if (targetFolder == null)
        {
            Debug.LogWarning("gears: Content folder not found; cannot import Gears.shader.");
            return;
        }

        Editor.Instance.ContentImporting.Import(sourcePath, targetFolder, skipSettingsDialog: true);
        Debug.Log("gears: importing Gears.shader to Content/Shaders/Gears.flax");
    }
}
#endif
