# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Validated telemetry storage with no-follow directory and file handles."""

import errno
import os
import stat
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import ClassVar


def _absolute_storage_path(path):
    path = Path(os.path.abspath(path))
    if sys.platform != "darwin" or len(path.parts) < 2 or path.parts[1] not in {"etc", "tmp", "var"}:
        return path
    root = Path(path.anchor)
    alias = root / path.parts[1]
    info = alias.lstat()
    if not stat.S_ISLNK(info.st_mode):
        return path
    root_info = root.stat()
    canonical = root / "private" / path.parts[1]
    target = Path(os.path.abspath(alias.parent / alias.readlink()))
    if info.st_uid != 0 or root_info.st_uid != 0 or root_info.st_mode & 0o022 or target != canonical:
        raise PermissionError("Telemetry storage cannot use an untrusted macOS system alias")
    # Resolve only the protected OS alias; user-controlled components still use no-follow opens.
    return canonical.joinpath(*path.parts[2:])


def _validate_file(info, allow_shared=False):
    if not stat.S_ISREG(info.st_mode):
        raise PermissionError("Telemetry storage requires a regular file")
    if getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        raise PermissionError("Telemetry storage cannot use reparse points")
    if not allow_shared and info.st_nlink != 1:
        raise PermissionError("Telemetry storage cannot use hard-linked files")
    if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
        raise PermissionError("Telemetry storage file is not owned by the current user")


def _validate_posix_directory(info, leaf=False):
    owners = {os.geteuid()} if leaf else {0, os.geteuid()}
    if info.st_uid not in owners:
        raise PermissionError("Telemetry storage directory is owned by another user")
    if not leaf and info.st_mode & 0o022 and not info.st_mode & stat.S_ISVTX:
        raise PermissionError("Telemetry storage ancestor is writable by another user")


def _validate_windows_owner(handle, kernel32):
    import ctypes
    import ctypes.wintypes as wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.GetSecurityInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetSecurityInfo.restype = wintypes.DWORD
    advapi32.OpenProcessToken.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    advapi32.EqualSid.restype = wintypes.BOOL
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p

    owner = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    token = wintypes.HANDLE()
    try:
        result = advapi32.GetSecurityInfo(handle, 1, 1, ctypes.byref(owner), None, None, None, ctypes.byref(descriptor))
        if result:
            raise ctypes.WinError(result)
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 8, ctypes.byref(token)):
            raise ctypes.WinError(ctypes.get_last_error())
        # An elevated token may create files owned by its default owner group.
        for info_class in (1, 4):  # TokenUser / TokenOwner
            size = wintypes.DWORD()
            advapi32.GetTokenInformation(token, info_class, None, 0, ctypes.byref(size))
            if not size.value:
                raise ctypes.WinError(ctypes.get_last_error())
            buffer = ctypes.create_string_buffer(size.value)
            if not advapi32.GetTokenInformation(token, info_class, buffer, size, ctypes.byref(size)):
                raise ctypes.WinError(ctypes.get_last_error())
            sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
            if advapi32.EqualSid(owner, sid):
                return
        raise PermissionError("Telemetry storage is not owned by the current user")
    finally:
        if token:
            kernel32.CloseHandle(token)
        if descriptor:
            kernel32.LocalFree(descriptor)


def _open_windows_path(path, flags=None):
    import ctypes
    import ctypes.wintypes as wintypes
    import msvcrt

    class AttributeTagInfo(ctypes.Structure):
        _fields_: ClassVar = [("attributes", wintypes.DWORD), ("tag", wintypes.DWORD)]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileInformationByHandleEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    directory = flags is None
    access = 0x00020080 if directory else 0x80000000  # READ_CONTROL | FILE_READ_ATTRIBUTES / GENERIC_READ
    if not directory and flags & (os.O_WRONLY | os.O_RDWR):
        access |= 0x40000000
    disposition = 3  # OPEN_EXISTING
    if not directory and flags & os.O_CREAT:
        disposition = 1 if flags & os.O_EXCL else 4  # CREATE_NEW / OPEN_ALWAYS
    attributes = 0x00200000 | (0x02000000 if directory else 0)  # OPEN_REPARSE_POINT / BACKUP_SEMANTICS
    # Omitting FILE_SHARE_DELETE pins the path while its handle is in use.
    handle = kernel32.CreateFileW(str(path), access, 3, None, disposition, attributes, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        info = AttributeTagInfo()
        if not kernel32.GetFileInformationByHandleEx(handle, 9, ctypes.byref(info), ctypes.sizeof(info)):
            raise ctypes.WinError(ctypes.get_last_error())
        if info.attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise PermissionError("Telemetry storage cannot use reparse points")
        if bool(info.attributes & stat.FILE_ATTRIBUTE_DIRECTORY) != directory:
            raise PermissionError("Telemetry storage path has an unexpected file type")
        if directory:
            result = handle, kernel32
            handle = None
            return result
        _validate_windows_owner(handle, kernel32)
        fd = msvcrt.open_osfhandle(handle, flags & (os.O_RDONLY | os.O_WRONLY | os.O_RDWR) | os.O_BINARY)
        handle = None
        return fd
    finally:
        if handle is not None:
            kernel32.CloseHandle(handle)


class StorageDirectory:
    """Pin validated directory components while telemetry accesses their files."""

    def __init__(self, path, create=False):
        self.path = _absolute_storage_path(path)
        if self.path == Path(self.path.anchor):
            raise PermissionError("Telemetry storage requires a per-user directory")
        self._create = create
        self._fd = None
        self._windows_handles = []

    def __enter__(self):
        return self.open()

    def open(self):
        try:
            if os.name == "nt":
                current = Path(self.path.anchor)
                self._windows_handles.append(_open_windows_path(current))
                for component in self.path.parts[1:]:
                    current /= component
                    if self._create:
                        with suppress(FileExistsError):
                            current.mkdir(mode=0o700)
                    self._windows_handles.append(_open_windows_path(current))
                _validate_windows_owner(*self._windows_handles[-1])
            else:
                flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                self._fd = os.open(self.path.anchor, flags)
                for component in self.path.parts[1:]:
                    if self._create:
                        with suppress(FileExistsError):
                            os.mkdir(component, mode=0o700, dir_fd=self._fd)
                    next_fd = os.open(component, flags, dir_fd=self._fd)
                    os.close(self._fd)
                    self._fd = next_fd
                    _validate_posix_directory(os.fstat(self._fd))
                _validate_posix_directory(os.fstat(self._fd), leaf=True)
                os.fchmod(self._fd, 0o700)
            return self
        except Exception as error:
            self.close()
            if isinstance(error, OSError) and error.errno in {errno.ELOOP, errno.ENOTDIR}:
                raise PermissionError("Telemetry storage directory cannot follow symlinks") from error
            raise

    def __exit__(self, *_args):
        self.close()

    def close(self):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        for handle, kernel32 in reversed(self._windows_handles):
            kernel32.CloseHandle(handle)
        self._windows_handles.clear()

    def fileno(self):
        if self._fd is None:
            raise ValueError("Telemetry storage directory has no open POSIX descriptor")
        return self._fd

    def validate_file(self, name):
        try:
            info = (
                os.stat(name, dir_fd=self._fd, follow_symlinks=False)
                if self._fd is not None
                else (self.path / name).lstat()
            )
        except FileNotFoundError:
            return
        _validate_file(info)
        if self._fd is None:
            fd = self.open_file(name, os.O_RDONLY)
            os.close(fd)

    def open_file(self, name, flags, allow_shared=False):
        if self._fd is not None:
            fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=self._fd)
        else:
            fd = _open_windows_path(self.path / name, flags)
        try:
            _validate_file(os.fstat(fd), allow_shared)
            if self._fd is not None and not allow_shared:
                os.fchmod(fd, 0o600)
            return fd
        except Exception:
            os.close(fd)
            raise

    def verify_file_handle(self, name, fd):
        with StorageDirectory(self.path) as current:
            current.validate_file(name)
            if self._fd is not None:
                if not os.path.samestat(os.fstat(self._fd), os.fstat(current.fileno())):
                    raise PermissionError("Telemetry storage directory changed during initialization")
                info = os.stat(name, dir_fd=self._fd, follow_symlinks=False)
                if not os.path.samestat(info, os.fstat(fd)):
                    raise PermissionError("Telemetry storage file changed during initialization")

    def publish(self, temporary_name, name, replace):
        self.validate_file(name)
        if self._fd is not None:
            if replace:
                os.replace(temporary_name, name, src_dir_fd=self._fd, dst_dir_fd=self._fd)
            else:
                os.link(temporary_name, name, src_dir_fd=self._fd, dst_dir_fd=self._fd, follow_symlinks=False)
        elif replace:
            deadline = time.monotonic() + 1.0
            while True:
                try:
                    (self.path / temporary_name).replace(self.path / name)
                    break
                except OSError as error:
                    # A concurrent validated reader pins the inode against replacement on Windows.
                    if getattr(error, "winerror", None) not in {5, 32, 33} or time.monotonic() >= deadline:
                        raise
                    time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
        else:
            os.link(self.path / temporary_name, self.path / name)

    def unlink(self, name):
        if self._fd is not None:
            os.unlink(name, dir_fd=self._fd)
        else:
            (self.path / name).unlink()
