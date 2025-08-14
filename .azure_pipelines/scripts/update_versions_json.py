import json
import sys


def main():
    """Add new version to versions.json."""
    if len(sys.argv) != 2:
        print("Usage: python update_versions_json.py <version>")
        sys.exit(1)
    
    new_version = sys.argv[1]
    # Path relative to gh-pages branch root where the main docs are
    versions_file = "_static/versions.json"
    
    # Read existing versions.json
    with open(versions_file, 'r') as f:
        versions = json.load(f)
    
    # Check if version already exists
    for v in versions:
        if v["version"] == new_version:
            print(f"Version {new_version} already exists in versions.json")
            return
    
    # Remove "(latest)" from all existing versions
    for v in versions:
        if "(latest)" in v["name"]:
            v["name"] = v["version"]
    
    # Insert new version after "dev (main)" with (latest) tag
    new_entry = {
        "name": f"{new_version} (latest)",
        "version": new_version,
        "url": f"https://microsoft.github.io/Olive/{new_version}/"
    }
    
    # Insert after the first entry (dev/main)
    versions.insert(1, new_entry)
    
    # Write updated versions.json
    with open(versions_file, 'w') as f:
        json.dump(versions, f, indent=4)
        f.write('\n')
    
    print(f"Added version {new_version} to versions.json")


if __name__ == "__main__":
    main()