#!/usr/bin/env python
import importlib.util
import sys, os

def run(path_from_ws, file):
    ws_dir = "/".join(os.path.abspath(__file__).split('/')[:-5])
    path=f'/{ws_dir}/{path_from_ws}/'
    os.chdir(path)
    sys.path.append(path)

    spec = importlib.util.spec_from_file_location(file, path+file)
    foo = importlib.util.module_from_spec(spec)
    sys.modules[file] = foo
    spec.loader.exec_module(foo)
    foo.main()
