#if FLAX_EDITOR
using System.IO;
using FlaxEditor;
using FlaxEditor.Content;
using FlaxEngine;

/// <summary>
/// Imports Triangle.shader into Content on editor load.
/// </summary>
public class TriangleEditorPlugin : EditorPlugin
{
    /// <inheritdoc />
    public override void InitializeEditor()
    {
        base.InitializeEditor();
        EnsureShaderAsset();
    }

    private static void EnsureShaderAsset()
    {
        var sourcePath = Path.Combine(Globals.ProjectFolder, "Source", "Shaders", "Triangle.shader");
        if (!File.Exists(sourcePath))
            return;

        var outputDir = Path.Combine(Globals.ProjectContentFolder, "Shaders");
        var outputPath = Path.Combine(outputDir, "Triangle.flax");
        Directory.CreateDirectory(outputDir);

        var needsImport = !File.Exists(outputPath)
            || File.GetLastWriteTimeUtc(sourcePath) > File.GetLastWriteTimeUtc(outputPath);

        if (!needsImport)
            return;

        var targetFolder = Editor.Instance.ContentDatabase.Find(outputDir) as ContentFolder
            ?? Editor.Instance.ContentDatabase.Find(Globals.ProjectContentFolder) as ContentFolder;
        if (targetFolder == null)
        {
            Debug.LogWarning("triangle: Content folder not found; cannot import Triangle.shader.");
            return;
        }

        Editor.Instance.ContentImporting.Import(sourcePath, targetFolder, skipSettingsDialog: true);
        Debug.Log("triangle: importing Triangle.shader to Content/Shaders/Triangle.flax");
    }
}
#endif
