import pandas as pd

def extract_model_family(model):

    model = str(model).strip()

    # Multiword model families
    multiword_families = [
        'Range Rover Sport',
        'Range Rover Evoque',
        'Range Rover Velar',
        'Range Rover',
        'Corolla Cross',
        'Urban Cruiser',
        'Land Cruiser Prado',
        'Land Cruiser',
        'Tiguan Allspace',
        'Grand Cherokee',
        'G-Class',
        'GL-Class',
        'M-Class',
        'T-Cross',
        'T-Roc',
        'CX-3',
        'CX-5',
        'CX-30',
        'CX-60',
        'CX-9'
    ]

    # Match longer names first
    for family in sorted(
        multiword_families,
        key=len,
        reverse=True
    ):
        if (
            model.casefold() == family.casefold()
            or model.casefold().startswith(
                family.casefold() + ' '
            )
        ):
            return family

    # Default: first word of model description
    return model.split()[0]
