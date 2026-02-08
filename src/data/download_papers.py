import sys
from pathlib import Path

# Adding project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.acl_loader import ACLAnthologyLoader

project_root = Path(__file__).parent.parent.parent
loader = ACLAnthologyLoader(data_dir=str(project_root / "data" / "acl"))

for venue in ["ACL", "EMNLP", "NAACL", "EACL", "COLING"]:
    print(f"\n--- Downloading {venue} 2025 ---")
    result = loader.download_by_venue(venue=venue, year=2025, limit=200, save_metadata=True)
    print(f"Result: {result}")
