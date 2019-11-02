import ctypes
# used for PyCapsule manipulation
if hasattr(ctypes, 'pythonapi'):
    ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object


def c_str(string):
    return ctypes.c_char_p(string.encode('utf-8'))


_c_str_dltensor = c_str('dltensor')
_c_str_used_dltensor = c_str('used_dltensor')


def to_capsule(ad_tensor):
    add = int(ad_tensor.numpy())
    ptr = ctypes.c_void_p(add)
    capsule = ctypes.pythonapi.PyCapsule_New(ptr, _c_str_dltensor, None)
    return capsule

