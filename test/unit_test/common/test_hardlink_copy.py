import random
import string
import sys

import pytest

from olive.common.utils import hardlink_copy_dir, hardlink_copy_file

randbytes = (
    random.randbytes if sys.version_info >= (3, 9) else lambda n: str(random.choice(string.printable) for _ in range(n))
)


def _randstr(start, stop):
    return str(randbytes(random.randint(start, stop)))


@pytest.fixture(name="create_dir")
def create_dir_fixture(tmp_path):
    src_dir = tmp_path / "src_dir"
    src_dir.mkdir(parents=True, exist_ok=True)
    sub_dirs = ["sub_dir1"]
    files = ["file1.ext1", "file2.ext2", "sub_dir1/file3.ext1", "sub_dir1/file4.ext2"]
    for sub_dir in sub_dirs:
        (src_dir / sub_dir).mkdir(parents=True, exist_ok=True)
    for file in files:
        with (src_dir / file).open("wt") as strm:
            strm.write(_randstr(1, 100))
    return src_dir


def test_copy_file_to_dir(create_dir, tmp_path):
    # setup
    src_dirpath = create_dir
    dst_dirpath = tmp_path / "dst_dir"

    filename = "file1.ext1"
    src_filepath = src_dirpath / filename
    dst_filepath = dst_dirpath / filename

    # test
    dst_dirpath.mkdir(parents=True)
    hardlink_copy_file(src_filepath, dst_dirpath)

    # assert
    assert dst_filepath.exists()
    assert dst_filepath.samefile(src_filepath)
    assert list(dst_dirpath.glob("**/*")) == [dst_filepath]


def test_copy_file_to_file(create_dir, tmp_path):
    # setup
    src_dirpath = create_dir
    dst_dirpath = tmp_path / "dst_dir"

    filename = "file1.ext1"
    src_filepath = src_dirpath / filename
    dst_filepath = dst_dirpath / filename

    # test
    dst_dirpath.mkdir(parents=True)
    hardlink_copy_file(src_filepath, dst_filepath)

    # assert
    assert dst_filepath.exists()
    assert dst_filepath.samefile(src_filepath)
    assert list(dst_dirpath.glob("**/*")) == [dst_filepath]


def test_copy_file_to_dir_overwrites(create_dir, tmp_path):
    # setup
    src_dirpath = create_dir
    dst_dirpath = tmp_path / "dst_dir"

    filename = "file1.ext1"
    src_filepath = src_dirpath / filename
    dst_filepath = dst_dirpath / filename

    dst_dirpath.mkdir(parents=True)
    with dst_filepath.open("wt") as strm:
        strm.write(_randstr(21, 30))

    assert dst_filepath.stat().st_size > 0

    # test
    hardlink_copy_file(src_filepath, dst_dirpath)

    # assert
    assert dst_filepath.exists()
    assert dst_filepath.samefile(src_filepath)
    assert dst_filepath.stat().st_size == src_filepath.stat().st_size
    assert list(dst_dirpath.glob("**/*")) == [dst_filepath]


def test_copy_dir_to_dir(create_dir, tmp_path):
    # setup
    src_dirpath = create_dir
    dst_dirpath = tmp_path / "dst_dir"

    # test
    hardlink_copy_dir(src_dirpath, dst_dirpath)

    # assert
    src_files = sorted(filepath.relative_to(src_dirpath) for filepath in src_dirpath.glob("**/*"))
    dst_files = sorted(filepath.relative_to(dst_dirpath) for filepath in dst_dirpath.glob("**/*"))
    assert src_files == dst_files

    for src_filepath in src_dirpath.glob("**/*"):
        dst_filepath = dst_dirpath / src_filepath.relative_to(src_dirpath)
        assert dst_filepath.exists()

        if src_filepath.is_file():
            assert dst_filepath.samefile(src_filepath)


def test_copy_dir_to_dir_overwrites(create_dir, tmp_path):
    # setup
    src_dirpath = create_dir
    dst_dirpath = tmp_path / "dst_dir"

    for src_filepath in src_dirpath.glob("**/*"):
        if src_filepath.is_file():
            dst_filepath = dst_dirpath / src_filepath.relative_to(src_dirpath)
            dst_filepath.parent.mkdir(parents=True, exist_ok=True)
            with dst_filepath.open("wt") as strm:
                strm.write(_randstr(101, 200))

            assert dst_filepath.stat().st_size > 0

    # test
    hardlink_copy_dir(src_dirpath, dst_dirpath)

    # assert
    src_files = sorted(filepath.relative_to(src_dirpath) for filepath in src_dirpath.glob("**/*"))
    dst_files = sorted(filepath.relative_to(dst_dirpath) for filepath in dst_dirpath.glob("**/*"))
    assert src_files == dst_files

    for src_filepath in src_dirpath.glob("**/*"):
        dst_filepath = dst_dirpath / src_filepath.relative_to(src_dirpath)
        assert dst_filepath.exists()

        if src_filepath.is_file():
            assert dst_filepath.samefile(src_filepath)
            assert dst_filepath.stat().st_size == src_filepath.stat().st_size


def test_del_src_keep_dst(create_dir, tmp_path):
    # setup
    src_dirpath = create_dir
    dst_dirpath = tmp_path / "dst_dir"

    filename = "file1.ext1"
    src_filepath = src_dirpath / filename
    dst_filepath = dst_dirpath / filename

    hardlink_copy_dir(src_dirpath, dst_dirpath)

    # test
    assert src_filepath.exists()
    assert dst_filepath.exists()
    assert src_filepath.samefile(dst_filepath)

    src_filepath.unlink()

    assert not src_filepath.exists()
    assert dst_filepath.exists()


def test_del_dst_keep_src(create_dir, tmp_path):
    # setup
    src_dirpath = create_dir
    dst_dirpath = tmp_path / "dst_dir"

    filename = "file1.ext1"
    src_filepath = src_dirpath / filename
    dst_filepath = dst_dirpath / filename

    hardlink_copy_dir(src_dirpath, dst_dirpath)

    # test
    assert src_filepath.exists()
    assert dst_filepath.exists()
    assert src_filepath.samefile(dst_filepath)

    dst_filepath.unlink()

    assert src_filepath.exists()
    assert not dst_filepath.exists()


def test_rename_has_no_impact(create_dir, tmp_path):
    # setup
    src_dirpath = create_dir
    dst_dirpath = tmp_path / "dst_dir"
    new_dst_dirpath = tmp_path / "new_dst_dir"

    hardlink_copy_dir(src_dirpath, dst_dirpath)

    dst_dirpath.rename(new_dst_dirpath)
    dst_dirpath = new_dst_dirpath

    # assert
    src_files = sorted(filepath.relative_to(src_dirpath) for filepath in src_dirpath.glob("**/*"))
    dst_files = sorted(filepath.relative_to(dst_dirpath) for filepath in dst_dirpath.glob("**/*"))
    assert src_files == dst_files

    for src_filepath in src_dirpath.glob("**/*"):
        dst_filepath = dst_dirpath / src_filepath.relative_to(src_dirpath)
        assert dst_filepath.exists()

        if src_filepath.is_file():
            assert dst_filepath.samefile(src_filepath)


if __name__ == "__main__":
    sys.exit(pytest.main(["-s", "test/unit_test/common/test_hardlink_copy.py"]))
