import os
DEBUG = os.environ.get("TORCHPCP_DEBUG")

if DEBUG == "true":
    # for debug
    from torch.utils.cpp_extension import load
    from setup import sources
    _backend = load(name='torchpcp_cpp', extra_cflags=['-O3', '-std=c++17'],
                    sources=sources)
else:
    # https://stackoverflow.com/questions/65710713/importerror-libc10-so-cannot-open-shared-object-file-no-such-file-or-director
    import torch
    from torchpcp.modules.functional.backend import cpp_ex
    _backend = cpp_ex
