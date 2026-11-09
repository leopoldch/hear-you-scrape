from pathlib import Path

import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
CLUSTER = ROOT / "data/cluster/cluster_haptic_texture_dataset"
LMT = ROOT / "data/lmt108/LMT_108_SurfaceMaterials_Database"

CLUSTER_TEXTILES_OUTSIDE_CLOTHS = [87, 88, 89]
CLUSTER_RIGID_PLASTICS = [49, 50, 51, 53, 54, 55, 56, 57, 58]
CLUSTER_ELASTOMERS = [63]

LMT_WOOD = [
    "G4Beech_0",
    "G4CherryTree",
    "G4CoreBeech",
    "G4Larch",
    "G4ProfiledWoodPlate",
    "G4Teak",
    "G4WoodTypeSilverOak",
]
LMT_METAL = ["G1ProfiledAluminumPlate", "G3AluminumPlate", "G3Brass", "G3Copper", "G3StainlessSteel"]
LMT_RIGID_PLASTICS = ["G3AcrylicGlass", "G3EpoxyPlate", "G8HighDensityPolyethylen"]
LMT_ELASTOMERS = [
    "G5FineRubber",
    "G5ProfiledRubberPlate",
    "G5RubberPlateVersion2",
    "G5SolidRubberPlateVersion1",
    "G5SolidRubberPlateVersion2",
]


SENS3_CATEGORIES = {
    "Fabric": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14, 15],
    "Sandpaper": [11, 12, 13, 16],
    "Vinyl": [17, 18, 19, 20],
    "Plastic": [21, 22],
    "Leather": [23, 24],
    "Rubber": [25],
    "Paper": [26, 27, 28, 29, 30, 31],
    "Metal": [32, 33, 34, 35, 36],
    "Wood": [37, 38, 39, 40, 41, 42],
    "Foam": [43, 44, 45, 46, 47, 48, 49, 50],
}
SENS3_FAMILY_OF_CATEGORY = {
    "Wood": ("wood", ""),
    "Metal": ("metal", ""),
    "Fabric": ("textile", ""),
    "Plastic": ("polymer", "rigid plastic"),
    "Rubber": ("polymer", "elastomer"),
    "Vinyl": ("polymer", "vinyl"),
}

# Penn Haptic Texture Toolkit (HaTT), categories of its Table 1. Same conventions as above:
# leather, foam, paper, coatings (painted wood, whiteboard) and unclear items are left out.
HATT_FAMILIES = {
    "wood": ["Wood", "Stained Wood", "MDF"],
    "metal": ["Aluminum", "Aluminum Foil", "Metal Mesh", "Metal Shelving", "Textured Metal"],
    "textile": [
        "Athletic Shirt",
        "Blanket",
        "Canvas 1",
        "Canvas 2",
        "Canvas 3",
        "Cotton",
        "Denim",
        "Felt",
        "Flannel",
        "Fleece",
        "Microfiber Cloth",
        "Nylon Bag",
        "Nylon Mesh",
        "Silk 1",
        "Silk 2",
        "Textured Cloth",
        "Towel",
        "Velvet",
        "Carpet 1",
        "Carpet 2",
        "Carpet 3",
        "Carpet 4",
    ],
    "polymer": [
        "ABS Plastic",
        "Binder",
        "File Portfolio",
        "Frosted Acrylic",
        "Plastic Mesh 1",
        "Plastic Mesh 2",
        "Tarp",
        "Wavy Acrylic",
        "Nitrile Glove",
        "Textured Rubber",
        "Vinyl 1",
        "Vinyl 2",
        "Pleather",
    ],
}
HATT_POLYMER_TYPE = {
    "Nitrile Glove": "elastomer",
    "Textured Rubber": "elastomer",
    "Vinyl 1": "vinyl",
    "Vinyl 2": "vinyl",
    "Pleather": "vinyl",
}


def hatt_family(name):
    for family, names in HATT_FAMILIES.items():
        if name in names:
            polymer_type = HATT_POLYMER_TYPE.get(name, "rigid plastic") if family == "polymer" else ""
            return family, polymer_type
    return "", ""


def cluster_family(number, name, category):
    if category == "Wood" and "laminated" not in name.lower():
        return "wood", ""
    if category == "Metals":
        return "metal", ""
    if (category == "Cloths" and name != "Mesh") or number in CLUSTER_TEXTILES_OUTSIDE_CLOTHS:
        return "textile", ""
    if number in CLUSTER_RIGID_PLASTICS:
        return "polymer", "rigid plastic"
    if number in CLUSTER_ELASTOMERS:
        return "polymer", "elastomer"
    return "", ""


def lmt_family(name):
    if name in LMT_WOOD:
        return "wood", ""
    if name in LMT_METAL:
        return "metal", ""
    if (name.startswith("G9") and "Leather" not in name) or name == "G6Towel":
        return "textile", ""
    if name in LMT_RIGID_PLASTICS:
        return "polymer", "rigid plastic"
    if name in LMT_ELASTOMERS:
        return "polymer", "elastomer"
    return "", ""


def surface_families():
    rows = []

    workbook = load_workbook(CLUSTER / "texture_list.xlsx", read_only=True, data_only=True)
    for number, name, _, category, *_ in list(workbook.active.values)[1:]:
        family, polymer_type = cluster_family(int(number), name, category)
        rows.append(
            {
                "dataset": "cluster",
                "surface": str(int(number)),
                "name": name,
                "native_category": category.strip(),
                "family": family,
                "polymer_type": polymer_type,
            }
        )
    workbook.close()

    lmt_names = sorted({path.name.split("_Sound_Movement_")[0] for path in LMT.rglob("*.wav")})
    for name in lmt_names:
        family, polymer_type = lmt_family(name)
        rows.append(
            {
                "dataset": "lmt108",
                "surface": name,
                "name": name[2:],
                "native_category": name[:2],
                "family": family,
                "polymer_type": polymer_type,
            }
        )

    for category, numbers in SENS3_CATEGORIES.items():
        family, polymer_type = SENS3_FAMILY_OF_CATEGORY.get(category, ("", ""))
        for number in numbers:
            rows.append(
                {
                    "dataset": "sens3",
                    "surface": str(number),
                    "name": f"{category} material {number}",
                    "native_category": category,
                    "family": family,
                    "polymer_type": polymer_type,
                }
            )

    hatt = ROOT / "data/hatt/raw"
    for name in sorted(path.name for path in hatt.iterdir() if path.is_dir()) if hatt.exists() else []:
        family, polymer_type = hatt_family(name)
        rows.append(
            {
                "dataset": "hatt",
                "surface": name,
                "name": name,
                "native_category": "",
                "family": family,
                "polymer_type": polymer_type,
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    families = surface_families()
    families.to_csv(ROOT / "analysis/surface_families.csv", index=False)
    kept = families[families.family != ""]
    print(kept.groupby(["dataset", "family", "polymer_type"]).size().to_string())
    print(f"\nexcluded: {(families.family == '').sum()} surfaces")
    print("saved", ROOT / "analysis/surface_families.csv")
