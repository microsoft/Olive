import json
import sys


def main():
    """Add new version to versions.json."""
    if len(sys.argv) != 2:
        sys.exit(1)

    new_version = sys.argv[1]
    # Path relative to gh-pages branch root where the main docs are
    versions_file = "_static/versions.json"

    # Read existing versions.json
    with open(versions_file) as f:
        versions = json.load(f)

    existing_entry = None

    # Remove "(latest)" and stale preferred flags from all existing versions.
    for v in versions:
        if "(latest)" in v["name"]:
            v["name"] = v["version"]
        v.pop("preferred", None)
        if v["version"] == new_version:
            existing_entry = v

    if existing_entry is None:
        existing_entry = {
            "name": f"{new_version} (latest)",
            "version": new_version,
            "url": f"https://microsoft.github.io/Olive/{new_version}/",
            "preferred": True,
        }
        # Insert after the first entry (dev/main)
        versions.insert(1, existing_entry)
    else:
        existing_entry["name"] = f"{new_version} (latest)"
        existing_entry["url"] = f"https://microsoft.github.io/Olive/{new_version}/"
        existing_entry["preferred"] = True

    # Keep the latest release directly after the dev/main entry.
    versions = [versions[0], existing_entry] + [v for v in versions[1:] if v is not existing_entry]

    # Write updated versions.json
    with open(versions_file, "w") as f:
        json.dump(versions, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    main()
