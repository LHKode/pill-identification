import argparse
import glob
import os
import string
import logging
from random import choice, randrange
from typing import List, Tuple
from PIL import Image
import tqdm
from pills_identification.workflows.pills_workflow import PillsWorkflow, PillsWorkflowStep


class GenerateImage(PillsWorkflowStep):
    size = (640, 640)
    scale = [0.2, 0.25, 0.3]

    def __init__(self, size: Tuple, scale: List[float]):
        self.size = size
        self.scale = scale

    def __call__(self, n_images: int, mode: str, bg_items: List[str], pill_items: List[str], output_path: str, **kwargs):
        return {**kwargs, "images": [self.generate_image(n_images, mode, bg_items, pill_items, output_path)]}

    @staticmethod
    def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
        """Random filename

        Args:
                size (int, optional): [description]. Defaults to 6.
                chars ([type], optional): [description]. Defaults to string.ascii_uppercase+string.digits.

        Returns:
                [type]: [description]
        """
        return "".join(choice(chars) for _ in range(size))

    @staticmethod
    def generate_result_folder(mode: str, output_path: str):
        """
            Generate folder to store images after generate

            Args:
                mode (str): train / valid / test
        """
        os.makedirs(f"{output_path}/images", exist_ok=True)
        os.makedirs(f"{output_path}/labels", exist_ok=True)
        os.makedirs(f"{output_path}/images/{mode}", exist_ok=True)
        os.makedirs(f"{output_path}/labels/{mode}", exist_ok=True)

    @staticmethod
    def scale_image(pill: Image, background: Image):
        """
            Scale pill and background to target size

            Args:
                pill (Image)
                background (Image)
        """
        # random scale
        pill_zoom = choice(GenerateImage.scale)
        pill_size = pill.size
        new_width = int(pill_size[0] * pill_zoom)
        new_height = int(pill_size[1] * pill_zoom)

        # Resize pill image
        pill = pill.resize((new_width, new_height))
        background = background.resize(GenerateImage.size)
        size = background.size

        # random position
        x = randrange(abs(size[0] - pill_size[0]))
        y = randrange(abs(size[1] - pill_size[1]))

        return x, y, size, background, pill

    @staticmethod
    def build_image(background: Image, pill_items: List[str], output_path: str, mode: str):
        """
            Build result image

            Args:
                background (Image): background image
                pill_items (List[str]): List pills
                output_path (str): target output folder
                mode (str): train / valid / test
        """
        num_of_pills = randrange(1, 4)
        name = GenerateImage.id_generator()

        try:
            background = Image.open(background).convert("RGBA")
            pills = [
                Image
                    .open(choice(pill_items))
                    .convert("RGBA")
                    .rotate(angle=randrange(0, 360), expand=True) 
                for _ in range(num_of_pills)
            ]
        except AttributeError as error:
            print(error)

        for i in range(num_of_pills):
            try:
                x, y, size, background, pills[i] = GenerateImage.scale_image(pills[i], background)

                w = pills[i].size[0]
                h = pills[i].size[1]

                # Add to background
                background.paste(pills[i], (x, y), mask=pills[i])

                # Write file .txt format yolov5
                if (x + w) > size[0]:
                    w = w - ((x + w) - size[0])

                if (y + h) > size[1]:
                    h = h - ((y + h) - size[1])

                x += w / 2
                y += h / 2

                x /= size[0]
                w /= size[0]

                y /= size[1]
                h /= size[1]
            except ValueError as error:
                logging.warning(error)
                continue

            with open(f"{output_path}/labels/{mode}/{name}.txt", mode="a", encoding="utf-8") as file:
                if i != num_of_pills - 1:
                    file.write(f"0 {x} {y} {w} {h} \n")
                else:
                    file.write(f"0 {x} {y} {w} {h}")

        background = background.convert('RGB')
        background.save(f"{output_path}/images/{mode}/{name}.jpg")


    @staticmethod
    def generate_image(n_images: int, mode: str, bg_items: List[str], pill_items: List[str], output_path: str):
        """Generate image data to train YOLO

        Args:
                n_images (int): Number of sample
                mode (string): train / valid / test
                bg_items (list[str]): list backgrounds
                pill_items (list[str]): list pills to add to background
                output_path (str): target output folder
        """

        GenerateImage.generate_result_folder(mode, output_path)

        n_iterators = tqdm.tqdm(range(n_images))

        for _ in n_iterators:
            background = choice(bg_items)
            try:
                GenerateImage.build_image(background, pill_items, output_path, mode)
            except AttributeError as error:
                logging.warning(error)
                continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_of_images", help="Number of images")
    parser.add_argument("--mode", help="Mode")
    parser.add_argument("--bg_path", help="Background path")
    parser.add_argument("--pill_path", help="Pills path")
    parser.add_argument("--output_path", help="Output path")
    args = parser.parse_args()

    images = int(args.num_of_images)
    bg_paths = list(glob.glob(f"{args.bg_path}/*.jpg"))
    pill_paths = list(glob.glob(f"{args.pill_path}/*.png"))

    steps = [
        GenerateImage((640, 640), [0.2, 0.25, 0.3]),
    ]
    workflows = PillsWorkflow(steps)

    workflows(n_images=images, mode=args.mode, bg_items=bg_paths, pill_items=pill_paths, output_path=args.output_path)
