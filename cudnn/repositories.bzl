load("@rules_cudnn//:cudnn/cudnn.bzl", "cudnn_dependencies")

def _non_module_dependencies_impl(_ctx):
    cudnn_dependencies()

cudnn = module_extension(
    implementation = _non_module_dependencies_impl,
)
