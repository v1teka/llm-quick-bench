"""Тонкая обёртка ctypes над C API Intel OSPRay 3 (Apache 2.0): path tracer на CPU + OpenImageDenoise.

Бинарная сборка OSPRay берётся с GitHub (RenderKit/ospray, релиз 3.2.0) и кладётся в
~/.cache/qbench/ospray-3.2.0.x86_64.linux (или путь из QBENCH_OSPRAY). См. ensure_ospray().
"""
from __future__ import annotations

import ctypes as C
import os
import tarfile
import urllib.request

import numpy as np

VERSION = "3.2.0"
URL = f"https://github.com/RenderKit/ospray/releases/download/v{VERSION}/ospray-{VERSION}.x86_64.linux.tar.gz"
DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), ".cache", "qbench", f"ospray-{VERSION}.x86_64.linux")

# значения перечислений OSPEnums.h (OSPRay 3.2)
T = dict(BOOL=250, DATA=134217828, CAMERA=134217829, GEOMETRIC_MODEL=134217832, GEOMETRY=134217833, GROUP=134217834,
         IMAGE_OPERATION=134217835, INSTANCE=134217836, LIGHT=134217837, MATERIAL=134217838, RENDERER=134217839,
         TEXTURE=134217840, WORLD=134217844, STRING=1500, UCHAR=2500, VEC3UC=2502, VEC4UC=2503, INT=4000, UINT=4500,
         VEC3UI=4502, VEC4UI=4503, FLOAT=6000, VEC2F=6001, VEC3F=6002, VEC4F=6003, AFFINE3F=12003)
TEX_RGBA8, TEX_SRGBA, TEX_RGB8, TEX_SRGB, TEX_R8, TEX_RGB32F = 0, 1, 3, 4, 6, 5
FB_RGBA32F, FB_SRGBA = 3, 2
FB_COLOR, FB_DEPTH, FB_ACCUM, FB_VARIANCE, FB_NORMAL, FB_ALBEDO, FB_ID_PRIM, FB_ID_OBJECT, FB_ID_INSTANCE = \
    1, 2, 4, 8, 16, 32, 64, 128, 256
DENOISE_LOW, DENOISE_MEDIUM, DENOISE_HIGH = 4, 5, 6


def ensure_ospray(root=None, log=print):
    """Путь к распакованному OSPRay; при отсутствии — скачать релиз с GitHub."""
    root = root or os.environ.get("QBENCH_OSPRAY") or DEFAULT_ROOT
    if os.path.exists(os.path.join(root, "lib", "libospray.so.3")):
        return root
    parent = os.path.dirname(root)
    os.makedirs(parent, exist_ok=True)
    tgz = os.path.join(parent, f"ospray-{VERSION}.tar.gz")
    if not os.path.exists(tgz):
        log(f"  скачиваю OSPRay {VERSION} (~130 МБ) с GitHub…")
        urllib.request.urlretrieve(URL, tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(parent)
    return root


class OSPRay:
    """Минимальный набор функций C API. Объекты — сырые указатели (int)."""

    def __init__(self, root=None, threads=None, log=print):
        root = ensure_ospray(root, log)
        lib_dir = os.path.join(root, "lib")
        # зависимости и модули заранее грузим глобально: dlopen модулей по имени их найдёт
        for name in ("libtbb.so.12", "libtbbmalloc.so.2", "librkcommon.so.1", "libembree4.so.4", "libopenvkl.so.2",
                     "libOpenImageDenoise_core.so.2.3.0", "libOpenImageDenoise.so.2", "libospray.so.3"):
            p = os.path.join(lib_dir, name)
            if os.path.exists(p):
                C.CDLL(p, mode=C.RTLD_GLOBAL)
        self.lib = L = C.CDLL(os.path.join(lib_dir, "libospray.so.3"), mode=C.RTLD_GLOBAL)
        vp, cp, u64, f32, i32, u32 = C.c_void_p, C.c_char_p, C.c_uint64, C.c_float, C.c_int, C.c_uint32
        sig = {
            "ospNewDevice": (vp, [cp]), "ospDeviceSetParam": (None, [vp, cp, u32, vp]), "ospDeviceCommit": (None, [vp]),
            "ospSetCurrentDevice": (None, [vp]), "ospLoadModule": (i32, [cp]),
            "ospDeviceGetLastErrorMsg": (cp, [vp]),
            "ospNewSharedData1D": (vp, [vp, u32, u64]), "ospNewSharedData2D": (vp, [vp, u32, u64, u64]),
            "ospNewData1D": (vp, [u32, u64]), "ospNewData2D": (vp, [u32, u64, u64]),
            "ospCopyData1D": (None, [vp, vp, u64]), "ospCopyData2D": (None, [vp, vp, u64, u64]),
            "ospNewGeometry": (vp, [cp]), "ospNewMaterial": (vp, [cp]), "ospNewGeometricModel": (vp, [vp]),
            "ospNewGroup": (vp, []), "ospNewInstance": (vp, [vp]), "ospNewWorld": (vp, []), "ospNewLight": (vp, [cp]),
            "ospNewCamera": (vp, [cp]), "ospNewRenderer": (vp, [cp]), "ospNewTexture": (vp, [cp]),
            "ospNewFrameBuffer": (vp, [i32, i32, u32, u32]), "ospNewImageOperation": (vp, [cp]),
            "ospSetString": (None, [vp, cp, cp]), "ospSetObject": (None, [vp, cp, vp]), "ospSetBool": (None, [vp, cp, i32]),
            "ospSetFloat": (None, [vp, cp, f32]), "ospSetInt": (None, [vp, cp, i32]), "ospSetUInt": (None, [vp, cp, u32]),
            "ospSetVec2f": (None, [vp, cp, f32, f32]), "ospSetVec3f": (None, [vp, cp, f32, f32, f32]),
            "ospSetVec4f": (None, [vp, cp, f32, f32, f32, f32]), "ospSetParam": (None, [vp, cp, u32, vp]),
            "ospSetObjectAsData": (None, [vp, cp, u32, vp]), "ospRemoveParam": (None, [vp, cp]),
            "ospCommit": (None, [vp]), "ospRelease": (None, [vp]), "ospRetain": (None, [vp]),
            "ospRenderFrameBlocking": (f32, [vp, vp, vp, vp]), "ospMapFrameBuffer": (vp, [vp, u32]),
            "ospUnmapFrameBuffer": (None, [vp, vp]), "ospResetAccumulation": (None, [vp]),
        }
        for name, (res, args) in sig.items():
            fn = getattr(L, name)
            fn.restype, fn.argtypes = res, args
        self.keep = []  # numpy-массивы, на которые ссылаются shared data
        if L.ospLoadModule(b"cpu") != 0:
            raise RuntimeError("OSPRay: не загрузился модуль cpu")
        self.has_denoiser = L.ospLoadModule(b"denoiser") == 0
        dev = L.ospNewDevice(b"cpu")
        if threads:
            v = C.c_int(int(threads))
            L.ospDeviceSetParam(dev, b"numThreads", T["INT"], C.byref(v))
        L.ospDeviceCommit(dev)
        L.ospSetCurrentDevice(dev)
        self.dev = dev

    # ---- данные
    def data(self, arr, typ):
        """Копия массива во владении OSPRay (тип T[...]). arr: (n, k) или (n,) для 1D, (h, w, k) для 2D."""
        L = self.lib
        a = np.ascontiguousarray(arr)
        if typ in (T["VEC4UC"], T["VEC3UC"]) and a.ndim == 3:
            h, w = a.shape[:2]
            sh = L.ospNewSharedData2D(a.ctypes.data, typ, w, h)
            d = L.ospNewData2D(typ, w, h)
            L.ospCopyData2D(sh, d, 0, 0)
        elif a.ndim == 3:
            h, w = a.shape[:2]
            sh = L.ospNewSharedData2D(a.ctypes.data, typ, w, h)
            d = L.ospNewData2D(typ, w, h)
            L.ospCopyData2D(sh, d, 0, 0)
        else:
            n = a.shape[0]
            sh = L.ospNewSharedData1D(a.ctypes.data, typ, n)
            d = L.ospNewData1D(typ, n)
            L.ospCopyData1D(sh, d, 0)
        L.ospRelease(sh)
        L.ospCommit(d)
        return d

    def objects(self, handles, typ):
        arr = (C.c_void_p * len(handles))(*handles)
        L = self.lib
        sh = L.ospNewSharedData1D(C.cast(arr, C.c_void_p), typ, len(handles))
        d = L.ospNewData1D(typ, len(handles))
        L.ospCopyData1D(sh, d, 0)
        L.ospRelease(sh)
        L.ospCommit(d)
        return d

    def set(self, obj, **kw):
        """Удобная установка параметров по типу значения."""
        L = self.lib
        for k, v in kw.items():
            name = k.replace("__", ".").encode()
            if isinstance(v, bool):
                L.ospSetBool(obj, name, int(v))
            elif isinstance(v, int):
                L.ospSetInt(obj, name, v)
            elif isinstance(v, float):
                L.ospSetFloat(obj, name, v)
            elif isinstance(v, str):
                L.ospSetString(obj, name, v.encode())
            elif isinstance(v, (tuple, list, np.ndarray)) and len(v) in (2, 3, 4) and not isinstance(v[0], (tuple, list)):
                f = [float(x) for x in v]
                (L.ospSetVec2f if len(f) == 2 else L.ospSetVec3f if len(f) == 3 else L.ospSetVec4f)(obj, name, *f)
            elif isinstance(v, Obj):
                L.ospSetObject(obj, name, v.h)
            else:
                L.ospSetObject(obj, name, v)
        return obj

    def uint(self, obj, name, v):
        self.lib.ospSetUInt(obj, name.encode(), int(v))

    def affine(self, obj, name, M):
        """M: 3×4 (линейная часть по столбцам + перенос) -> affine3f."""
        a = (C.c_float * 12)(*[float(x) for x in np.asarray(M, np.float32).T.reshape(-1)])
        self.lib.ospSetParam(obj, name.encode(), T["AFFINE3F"], C.cast(a, C.c_void_p))

    def commit(self, obj):
        self.lib.ospCommit(obj)
        return obj

    def release(self, obj):
        if obj:
            self.lib.ospRelease(obj)

    # ---- текстура из картинки (H, W, 3|4) uint8, sRGB
    def texture(self, img, srgb=True, nearest=False):
        L = self.lib
        a = np.asarray(img)
        if a.dtype != np.uint8:
            a = np.clip(a * 255 + 0.5, 0, 255).astype(np.uint8)
        if a.ndim == 2:
            a = np.repeat(a[:, :, None], 3, 2)
        a = np.ascontiguousarray(a[::-1])  # OSPRay: первая строка — низ
        rgba = a.shape[2] == 4
        tex = L.ospNewTexture(b"texture2d")
        d = self.data(a, T["VEC4UC"] if rgba else T["VEC3UC"])
        L.ospSetObject(tex, b"data", d)
        L.ospRelease(d)
        self.uint(tex, "format", (TEX_SRGBA if rgba else TEX_SRGB) if srgb else (TEX_RGBA8 if rgba else TEX_RGB8))
        if nearest:
            self.uint(tex, "filter", 1)
        L.ospCommit(tex)
        return tex

    def map(self, fb, channel, dtype, shape):
        L = self.lib
        p = L.ospMapFrameBuffer(fb, channel)
        n = int(np.prod(shape))
        buf = (C.c_byte * (n * np.dtype(dtype).itemsize)).from_address(p)
        out = np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
        L.ospUnmapFrameBuffer(p, fb)
        return out


class Obj:  # маркер для set(): обёртка указателя
    def __init__(self, h):
        self.h = h
