import torch

libs = {
    "_C": torch.library.Library("_C", "IMPL"),
    "_moe_C": torch.library.Library("_moe_C", "IMPL"),
    "_vllm_fa3_C": torch.library.Library("_vllm_fa3_C", "IMPL"),
    "_C_cache_ops": torch.library.Library("_C_cache_ops", "IMPL"),
}


def patch_module_method(cls, method_name: str, new_method: callable, verbose=True):
    old_method = getattr(cls, method_name, None)
    setattr(cls, method_name, new_method)
    if verbose:
        cls_name = cls.__name__
        new_method_name = new_method.__name__
        print(f"Patched {cls_name}.{method_name} with FLAGGEMS {new_method_name}")

    # incase we need to revert the patch later
    return old_method


def patch_vllm_lib(lib_name, fn_name, fn, key, verbose=True):
    lib = libs.get(lib_name, None)
    if lib is None:
        raise ValueError(f"Library {lib_name} is not recognized.")

    lib.impl(fn_name, fn, key)

    if verbose:
        print(f"Patched torch.ops.{lib_name}.{fn_name} with FLAGGEMS {fn.__name__}")
