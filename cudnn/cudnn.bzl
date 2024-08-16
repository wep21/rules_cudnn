"""Dependencies for cudnn."""

def _is_linux(repository_ctx):
    return repository_ctx.os.name.startswith("linux")

def _is_windows(repository_ctx):
    return repository_ctx.os.name.startswith("windows")

def _try_symlink_file(repository_ctx, include_path, file, out_prefix):
    suffix = "/" + file
    if repository_ctx.path(include_path + suffix).exists:
        repository_ctx.symlink(include_path + suffix, out_prefix + suffix)

def _symlink_cudnn_v8_headers(repository_ctx, include_path, out_prefix):
    for name in [
        "cudnn.h",
        "cudnn_adv_infer.h",
        "cudnn_adv_train.h",
        "cudnn_backend.h",
        "cudnn_cnn_infer.h",
        "cudnn_cnn_train.h",
        "cudnn_ops_infer.h",
        "cudnn_ops_train.h",
        "cudnn_version.h",
    ]:
        _try_symlink_file(repository_ctx, include_path, name, out_prefix)

def _local_cudnn_impl(repository_ctx):
    # Path to cudnn is
    # - taken from CUDNN_INSTALL_PATH environment variable or
    # - defaults to '/usr'
    cudnn_include_path = None
    cudnn_library_path = None

    if _is_linux(repository_ctx):
        cudnn_install_path = repository_ctx.os.environ.get("CUDNN_INSTALL_PATH", "/usr")
        if repository_ctx.path(cudnn_install_path).exists:
            cudnn_include_path = cudnn_install_path + "/include"
            if not repository_ctx.path(cudnn_include_path + "/cudnn.h").exists:
                cudnn_include_path = None

            for suffix in ["/lib/x86_64-linux-gnu", "/lib64", "/lib"]:
                if repository_ctx.path(cudnn_install_path + suffix + "/libcudnn.so").exists:
                    cudnn_library_path = cudnn_install_path + suffix
                    break
    elif _is_windows(repository_ctx):
        cuda_path = repository_ctx.os.environ.get("CUDA_PATH")
        cudnn_install_path = repository_ctx.os.environ.get("CUDNN_INSTALL_PATH", cuda_path)
        if repository_ctx.path(cudnn_install_path).exists:
            cudnn_include_path = cudnn_install_path + "/include"
            cudnn_library_path = cudnn_install_path + "/lib"
    else:
        fail("Unsupported OS")

    if (
        cudnn_include_path and repository_ctx.path(cudnn_include_path).exists and
	cudnn_library_path and repository_ctx.path(cudnn_library_path).exists
    ):
        repository_ctx.symlink(Label("//:cudnn/BUILD.local_cudnn"), "BUILD")
        _symlink_cudnn_v8_headers(repository_ctx, cudnn_include_path, "cudnn/include")
        repository_ctx.symlink(cudnn_library_path, "cudnn/lib")

    else:
        repository_ctx.file("BUILD", content = "")  # Empty file

_local_cudnn = repository_rule(
    implementation = _local_cudnn_impl,
    environ = ["CUDNN_INSTALL_PATH", "CUDA_PATH"],
    # remotable = True,
)

def cudnn_dependencies():
    _local_cudnn(name = "local_cudnn")
