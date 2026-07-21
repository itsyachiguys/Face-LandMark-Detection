import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(image_size=224):

    return A.Compose(
        [
            A.Resize(image_size, image_size),

            A.Normalize(
                mean=(0.5,),
                std=(0.5,),
            ),

            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
    )


def get_test_transforms(image_size=224):

    return A.Compose(
        [
            A.Resize(image_size, image_size),

            A.Normalize(
                mean=(0.5,),
                std=(0.5,),
            ),

            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            remove_invisible=False,
        ),
    )