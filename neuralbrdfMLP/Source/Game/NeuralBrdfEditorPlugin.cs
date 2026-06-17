#if FLAX_EDITOR
using System.IO;
using FlaxEditor;
using FlaxEditor.Content;
using FlaxEditor.CustomEditors;
using FlaxEditor.CustomEditors.Elements;
using FlaxEngine;
using FlaxEngine.GUI;

/// <summary>
/// Editor plugin for the Neural BRDF sample:
///  - imports the HLSL shaders into Content on editor load,
///  - adds a "Neural BRDF" entry to the Tools menu that opens a control window
///    with the ported RTXNS buttons (Train / Reset / Save / Load) plus material sliders.
/// </summary>
public class NeuralBrdfEditorPlugin : EditorPlugin
{
    private static readonly string[] Shaders = { "NeuralBrdf.shader", "NeuralBrdfTraining.shader" };
    private NeuralBrdfControlWindow _window;
    private FlaxEditor.GUI.ContextMenu.ContextMenuButton _menuButton;

    /// <inheritdoc />
    public override void InitializeEditor()
    {
        base.InitializeEditor();
        EnsureShaderAssets();

        // Guard against duplicate entries: InitializeEditor runs again on every script
        // hot-reload, so reuse the existing button instead of adding another one.
        _menuButton = Editor.Instance.UI.MenuTools.ContextMenu.AddButton("Neural BRDF", ShowWindow);
    }

    /// <inheritdoc />
    public override void DeinitializeEditor()
    {
        _menuButton?.Dispose();
        _menuButton = null;
        _window?.Window?.Close();
        _window = null;

        base.DeinitializeEditor();
    }

    private void ShowWindow()
    {
        if (_window == null)
            _window = new NeuralBrdfControlWindow();
        _window.Show();
    }

    private static void EnsureShaderAssets()
    {
        var outputDir = Path.Combine(Globals.ProjectContentFolder, "Shaders");
        Directory.CreateDirectory(outputDir);

        ContentFolder targetFolder = Editor.Instance.ContentDatabase.Find(outputDir) as ContentFolder
            ?? Editor.Instance.ContentDatabase.Find(Globals.ProjectContentFolder) as ContentFolder;
        if (targetFolder == null)
        {
            Debug.LogWarning("neuralbrdf: Content folder not found; cannot import shaders.");
            return;
        }

        foreach (var shader in Shaders)
        {
            var sourcePath = Path.Combine(Globals.ProjectFolder, "Source", "Shaders", shader);
            if (!File.Exists(sourcePath))
                continue;

            var outputPath = Path.Combine(outputDir, Path.GetFileNameWithoutExtension(shader) + ".flax");
            var needsImport = !File.Exists(outputPath)
                || File.GetLastWriteTimeUtc(sourcePath) > File.GetLastWriteTimeUtc(outputPath);
            if (!needsImport)
                continue;

            Editor.Instance.ContentImporting.Import(sourcePath, targetFolder, skipSettingsDialog: true);
            Debug.Log("neuralbrdf: importing " + shader);
        }
    }
}

/// <summary>
/// Control window mirroring the RTXNS UIWidget: training controls, model persistence and material sliders.
/// </summary>
public class NeuralBrdfControlWindow : CustomEditorWindow
{
    private Button _trainButton;
    private LabelElement _statsLabel;

    /// <inheritdoc />
    public override void Initialize(LayoutElementsContainer layout)
    {
        var group = layout.Group("Neural BRDF");

        _trainButton = group.Button("Start Training").Button;
        _trainButton.Clicked += () =>
        {
            var inst = NeuralBrdfRenderer.Instance;
            if (inst != null)
                inst.SetTrainingEnabled(!inst.TrainingEnabled);
        };

        group.Button("Reset").Button.Clicked += () =>
        {
            NeuralBrdfRenderer.Instance?.ResetTraining();
        };

        group.Button("Save Model...").Button.Clicked += () =>
        {
            var inst = NeuralBrdfRenderer.Instance;
            if (inst == null)
                return;
            if (FileSystem.ShowSaveFileDialog(Editor.Instance.Windows.MainWindow, Globals.ProjectFolder,
                    "Neural BRDF model (*.nbrd)\0*.nbrd\0All files (*.*)\0*.*\0", false, "Save trained model", out var files))
                return;
            if (files != null && files.Length > 0)
            {
                var path = files[0];
                if (!path.EndsWith(".nbrd"))
                    path += ".nbrd";
                inst.SaveModel(path);
            }
        };

        group.Button("Load Model...").Button.Clicked += () =>
        {
            var inst = NeuralBrdfRenderer.Instance;
            if (inst == null)
                return;
            if (FileSystem.ShowOpenFileDialog(Editor.Instance.Windows.MainWindow, Globals.ProjectFolder,
                    "Neural BRDF model (*.nbrd)\0*.nbrd\0All files (*.*)\0*.*\0", false, "Load trained model", out var files))
                return;
            if (files != null && files.Length > 0)
                inst.LoadModel(files[0]);
        };

        _statsLabel = group.Label("Enter Play mode to start training.");

        // Material / light controls (ported from the RTXNS sliders).
        AddSlider(group, "Light Intensity", 0.0f, 10.0f,
            () => NeuralBrdfRenderer.Instance?.LightIntensity ?? 3.0f,
            v => { var i = NeuralBrdfRenderer.Instance; if (i != null) i.LightIntensity = v; });
        AddSlider(group, "Specular", 0.0f, 1.0f,
            () => NeuralBrdfRenderer.Instance?.Specular ?? 0.5f,
            v => { var i = NeuralBrdfRenderer.Instance; if (i != null) i.Specular = v; });
        AddSlider(group, "Roughness", 0.0f, 1.0f,
            () => NeuralBrdfRenderer.Instance?.Roughness ?? 0.5f,
            v => { var i = NeuralBrdfRenderer.Instance; if (i != null) i.Roughness = v; });
        AddSlider(group, "Metallic", 0.0f, 1.0f,
            () => NeuralBrdfRenderer.Instance?.Metallic ?? 0.0f,
            v => { var i = NeuralBrdfRenderer.Instance; if (i != null) i.Metallic = v; });
    }

    private static void AddSlider(LayoutElementsContainer layout, string name, float min, float max,
        System.Func<float> get, System.Action<float> set)
    {
        var slider = layout.Slider(name);
        slider.Slider.SetLimits(new LimitAttribute(min, max));
        slider.Value = get();
        slider.Slider.ValueChanged += () => set(slider.Value);
    }

    /// <inheritdoc />
    public override void Refresh()
    {
        base.Refresh();

        var inst = NeuralBrdfRenderer.Instance;
        if (_trainButton != null)
            _trainButton.Text = (inst != null && inst.TrainingEnabled) ? "Pause Training" : "Start Training";
        if (_statsLabel?.Label != null)
        {
            _statsLabel.Label.Text = inst != null
                ? string.Format("Steps: {0}    Loss: {1:F5}", inst.Epochs, inst.Loss)
                : "Enter Play mode to start training.";
        }
    }
}
#endif
