using System.IO;
using Flax.Build;
using Flax.Build.NativeCpp;

public class Game : GameModule
{
    /// <inheritdoc />
    public override void Init()
    {
        base.Init();
        BuildNativeCode = true;
    }

    /// <inheritdoc />
    public override void Setup(BuildOptions options)
    {
        base.Setup(options);
        options.ScriptingAPI.IgnoreMissingDocumentationWarnings = true;

        var common = Path.Combine(Globals.Project.ProjectFolderPath, "..", "common");
        options.PublicIncludePaths.Add(common);
        var glm = Path.Combine(common, "glm");
        options.PublicIncludePaths.Add(glm);

        options.SourceFiles.Add(Path.Combine(common, "SaschaCamera.cpp"));
    }
}
