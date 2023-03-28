import sys
import hashlib
def sha256_checksum(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as file:
        for chunk in iter(lambda: file.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py zip_file_path")
        sys.exit(1)

    zip_file_path = sys.argv[1]
    checksum = sha256_checksum(zip_file_path)
    assert(checksum == "5a31da2221fdad3bb1312d46e1201cb7a3876066396897091bfed0ce459a4146")
    print(f"SHA-256 checksum of '{zip_file_path}': {checksum}")
