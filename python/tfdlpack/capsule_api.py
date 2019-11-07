# pylint: disable=invalid-name
"""Capsule API"""
import ctypes
# used for PyCapsule manipulation
if hasattr(ctypes, 'pythonapi'):
    ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
    ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
    ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object


def c_str(string):
    """c_str type"""
    return ctypes.c_char_p(string.encode('utf-8'))


_c_str_dltensor = c_str('dltensor')
_c_str_used_dltensor = c_str('used_dltensor')

DLPackPyCapsuleDestructor = ctypes.CFUNCTYPE(None, ctypes.c_void_p)


def to_capsule(ad_tensor):
    """convert address tensor to capsule"""
    add = int(ad_tensor.numpy())
    ptr = ctypes.c_void_p(add)
    capsule = ctypes.pythonapi.PyCapsule_New(ptr, _c_str_dltensor, DLPackPyCapsuleDestructor(0))
    return capsule


def get_capsule_address(dl_cap, consume=False):
    """get address from capsule"""
    dltensor = ctypes.py_object(dl_cap)
    if ctypes.pythonapi.PyCapsule_IsValid(dltensor, _c_str_dltensor):
        ptr = ctypes.pythonapi.PyCapsule_GetPointer(dltensor, _c_str_dltensor)
        if consume:
            ctypes.pythonapi.PyCapsule_SetName(
                dltensor, _c_str_used_dltensor)
            ctypes.pythonapi.PyCapsule_SetDestructor(
                dltensor, DLPackPyCapsuleDestructor(0))
        return ptr
    raise ValueError(
        "Expect a dltensor field, PyCapsule can only be consumed once")
